#!/usr/bin/env python3
"""End-to-end TASFT validation: train -> export -> inference -> report.

Runs a complete validation pipeline on Qwen/Qwen2.5-0.5B using CPU-only
float32 training with synthetic medical QA data. Validates that the full
TASFT co-training loop (LoRA + AttnGate), bundle export, and inference
pipeline function correctly on Apple Silicon hardware.

Hardware target: Apple M4 Pro, 48GB RAM, CPU-only (no CUDA, no MPS for training).
Model: Qwen/Qwen2.5-0.5B (494M params, 24 layers, 14 heads, 2 KV heads, GQA).

Usage:
    python scripts/run_e2e_validation.py --output-dir /path/to/output

Preconditions:
    - tasft package installed in editable mode
    - transformers, peft, safetensors installed
    - Sufficient RAM (~8GB for model + training buffers)

Postconditions:
    - Training checkpoint saved to output_dir/checkpoints/
    - Bundle exported to output_dir/bundle/
    - validation_report.json written to output_dir/
    - Summary printed to stdout

Complexity: O(50 * forward_pass_cost) for training, O(5 * generate_cost) for inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import structlog
import torch
from safetensors.torch import save_file
from torch.utils.data import Dataset

# ------------------------------------------------------------------
# Structured logging setup
# ------------------------------------------------------------------
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)
log = structlog.get_logger("e2e_validation")

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
SEED: Final[int] = 42
MODEL_ID: Final[str] = "Qwen/Qwen2.5-0.5B"
NUM_LAYERS: Final[int] = 24
NUM_TRAIN_STEPS: Final[int] = 50
BATCH_SIZE: Final[int] = 1
MAX_SEQ_LEN: Final[int] = 128
BLOCK_SIZE: Final[int] = 64
LEARNING_RATE: Final[float] = 5e-5
LAMBDA_GATE: Final[float] = 0.1
BETA_SPARSE: Final[float] = 0.01
TAU_TARGET: Final[float] = 0.8
GATE_LR_RATIO: Final[float] = 0.1
# Gate warmup of 10 steps allows the task loss to stabilize before the gate
# distillation objective activates. After step 10, the full co-training
# objective L = L_task + lambda * L_gate is active for the remaining steps.
GATE_WARMUP_STEPS: Final[int] = 10
LAYERS_PER_STEP: Final[int] = 4
LORA_RANK: Final[int] = 8
LORA_ALPHA: Final[int] = 16
CHECKPOINT_INTERVAL: Final[int] = 25
NUM_SYNTHETIC_SAMPLES: Final[int] = 100
NUM_INFERENCE_PROMPTS: Final[int] = 5
MAX_NEW_TOKENS: Final[int] = 64

# ------------------------------------------------------------------
# Synthetic Medical QA Dataset
# ------------------------------------------------------------------
MEDICAL_QA_PAIRS: Final[list[tuple[str, str]]] = [
    (
        "What is the primary treatment for hypertension?",
        "First-line treatment includes lifestyle modifications and antihypertensive medications such as ACE inhibitors, ARBs, calcium channel blockers, or thiazide diuretics. Blood pressure targets are typically below 130/80 mmHg.",
    ),
    (
        "What are the symptoms of type 2 diabetes mellitus?",
        "Common symptoms include polyuria, polydipsia, polyphagia, unexplained weight loss, fatigue, blurred vision, and slow wound healing. Many patients are asymptomatic and diagnosed on routine screening.",
    ),
    (
        "How is acute myocardial infarction diagnosed?",
        "Diagnosis requires a rise and fall in cardiac troponin with at least one of: ischemic symptoms, new ST-segment or T-wave changes, pathological Q waves, imaging evidence of myocardial loss, or intracoronary thrombus.",
    ),
    (
        "What is the recommended treatment for community-acquired pneumonia?",
        "Outpatient treatment typically includes amoxicillin or a macrolide like azithromycin. Inpatients receive a beta-lactam plus a macrolide, or a respiratory fluoroquinolone monotherapy.",
    ),
    (
        "What are the risk factors for deep vein thrombosis?",
        "Major risk factors include immobility, recent surgery, malignancy, pregnancy, oral contraceptives, inherited thrombophilias such as Factor V Leiden, obesity, and advanced age.",
    ),
    (
        "How is chronic kidney disease staged?",
        "CKD is staged by GFR: Stage 1 is GFR 90 or more with kidney damage markers, Stage 2 is 60 to 89, Stage 3a is 45 to 59, Stage 3b is 30 to 44, Stage 4 is 15 to 29, and Stage 5 is below 15 requiring dialysis.",
    ),
    (
        "What is the pathophysiology of asthma?",
        "Asthma involves chronic airway inflammation with bronchial hyperresponsiveness, leading to reversible airflow obstruction. Key features include mast cell degranulation, eosinophilic infiltration, mucus hypersecretion, and smooth muscle hypertrophy.",
    ),
    (
        "What are the first-line treatments for major depressive disorder?",
        "First-line pharmacotherapy includes SSRIs such as sertraline, escitalopram, or fluoxetine. Cognitive behavioral therapy is equally effective and recommended as monotherapy or combined with medication.",
    ),
    (
        "How is rheumatoid arthritis diagnosed?",
        "Diagnosis uses the 2010 ACR/EULAR criteria including joint involvement, serology such as RF and anti-CCP, acute phase reactants like ESR and CRP, and duration of symptoms greater than six weeks.",
    ),
    (
        "What is the management of acute appendicitis?",
        "Standard treatment is appendectomy, either laparoscopic or open. Non-operative management with antibiotics may be considered for uncomplicated cases. Perioperative antibiotics reduce surgical site infections.",
    ),
    (
        "What are the clinical features of Graves disease?",
        "Features include diffuse goiter, exophthalmos, pretibial myxedema, tachycardia, weight loss, heat intolerance, tremor, and anxiety. TSH receptor antibodies are elevated with suppressed TSH.",
    ),
    (
        "How is iron deficiency anemia treated?",
        "Oral ferrous sulfate is first-line, typically 325mg daily. IV iron is used for malabsorption, intolerance, or severe anemia. Underlying cause must be identified, especially ruling out gastrointestinal bleeding.",
    ),
    (
        "What is the differential diagnosis for chest pain?",
        "Differential includes acute coronary syndrome, pulmonary embolism, aortic dissection, pneumothorax, esophageal rupture, pericarditis, costochondritis, gastroesophageal reflux, and musculoskeletal causes.",
    ),
    (
        "How is atrial fibrillation managed?",
        "Management involves rate control with beta-blockers or calcium channel blockers, rhythm control with antiarrhythmics or cardioversion, and stroke prevention with anticoagulation guided by CHA2DS2-VASc score.",
    ),
    (
        "What are the stages of chronic obstructive pulmonary disease?",
        "GOLD staging uses post-bronchodilator FEV1 percentage predicted: Stage 1 mild is 80 percent or more, Stage 2 moderate is 50 to 79, Stage 3 severe is 30 to 49, Stage 4 very severe is below 30.",
    ),
    (
        "What is the treatment for acute ischemic stroke?",
        "IV alteplase within 4.5 hours of onset is standard. Mechanical thrombectomy within 24 hours for large vessel occlusion. Aspirin within 24 to 48 hours. Blood pressure management and supportive care.",
    ),
    (
        "How is celiac disease diagnosed?",
        "Serologic testing with tissue transglutaminase IgA antibodies followed by duodenal biopsy showing villous atrophy, crypt hyperplasia, and intraepithelial lymphocytosis. HLA-DQ2 or DQ8 testing supports diagnosis.",
    ),
    (
        "What are the complications of diabetes mellitus?",
        "Microvascular complications include retinopathy, nephropathy, and neuropathy. Macrovascular complications include coronary artery disease, peripheral arterial disease, and cerebrovascular disease. Diabetic ketoacidosis is an acute complication.",
    ),
    (
        "How is congestive heart failure classified?",
        "NYHA classification: Class I has no limitation of physical activity, Class II has slight limitation, Class III has marked limitation, Class IV has symptoms at rest. Also classified as HFrEF or HFpEF by ejection fraction.",
    ),
    (
        "What is the management of sepsis?",
        "Hour-1 bundle includes measuring lactate, blood cultures, broad-spectrum antibiotics, 30 mL per kg crystalloid for hypotension, and vasopressors for MAP below 65 mmHg. Source control is essential.",
    ),
    (
        "What are the indications for insulin therapy in type 2 diabetes?",
        "Insulin is indicated when HbA1c remains above target despite maximal oral agents, for symptomatic hyperglycemia, during pregnancy, or for acute illness. Basal insulin is typically initiated first.",
    ),
    (
        "How is hypothyroidism treated?",
        "Levothyroxine is the standard replacement therapy, with dose titrated to normalize TSH. Starting dose is typically 1.6 mcg per kg per day, adjusted every 6 to 8 weeks. Take on empty stomach.",
    ),
    (
        "What is the pathophysiology of peptic ulcer disease?",
        "Caused by an imbalance between aggressive factors like acid and pepsin and protective factors like mucus and bicarbonate. H. pylori infection and NSAID use are the two main etiologies.",
    ),
    (
        "How is osteoporosis diagnosed and treated?",
        "Diagnosis by DEXA scan with T-score at or below minus 2.5. Treatment includes bisphosphonates like alendronate, calcium and vitamin D supplementation, weight-bearing exercise, and fall prevention.",
    ),
    (
        "What are the signs of meningitis?",
        "Classic triad of fever, neck stiffness, and altered mental status. Other signs include headache, photophobia, Kernig sign, Brudzinski sign, and petechial rash in meningococcal disease.",
    ),
    (
        "How is acute pancreatitis managed?",
        "Management includes aggressive IV fluid resuscitation, pain control, NPO status initially, and early enteral feeding when tolerated. ERCP for gallstone pancreatitis with choledocholithiasis.",
    ),
    (
        "What is the treatment for pulmonary embolism?",
        "Anticoagulation with heparin bridged to warfarin or direct oral anticoagulants. Massive PE with hemodynamic instability may require thrombolytics or surgical embolectomy. IVC filter if anticoagulation contraindicated.",
    ),
    (
        "How is Parkinsons disease treated?",
        "Levodopa combined with carbidopa is the gold standard. Dopamine agonists, MAO-B inhibitors, and COMT inhibitors are adjunctive. Deep brain stimulation for advanced cases with motor fluctuations.",
    ),
    (
        "What are the causes of acute kidney injury?",
        "Prerenal causes include hypovolemia and heart failure. Intrinsic causes include acute tubular necrosis, glomerulonephritis, and interstitial nephritis. Postrenal causes include urinary obstruction.",
    ),
    (
        "How is epilepsy managed?",
        "Antiepileptic drugs are first-line: levetiracetam, lamotrigine, or valproate depending on seizure type. Surgery may be considered for drug-resistant epilepsy with identifiable focus.",
    ),
    (
        "What is the approach to managing hyperkalemia?",
        "Stabilize myocardium with IV calcium gluconate, shift potassium intracellularly with insulin plus glucose and sodium bicarbonate, remove potassium with loop diuretics, sodium polystyrene, or dialysis.",
    ),
    (
        "How is gout treated?",
        "Acute attacks are treated with NSAIDs, colchicine, or corticosteroids. Chronic management includes urate-lowering therapy with allopurinol or febuxostat, targeting serum urate below 6 mg per dL.",
    ),
    (
        "What are the features of systemic lupus erythematosus?",
        "SLE presents with malar rash, photosensitivity, oral ulcers, arthritis, serositis, renal involvement, neurologic manifestations, hematologic abnormalities, and positive ANA and anti-dsDNA antibodies.",
    ),
    (
        "How is cirrhosis of the liver managed?",
        "Management includes treating underlying cause, surveillance for hepatocellular carcinoma, managing complications like ascites with diuretics, variceal bleeding prophylaxis with beta-blockers, and liver transplant evaluation.",
    ),
    (
        "What is the treatment for urinary tract infection?",
        "Uncomplicated cystitis is treated with nitrofurantoin, trimethoprim-sulfamethoxazole, or fosfomycin. Pyelonephritis requires fluoroquinolones or parenteral antibiotics. Complicated UTI needs broader coverage.",
    ),
    (
        "How is thyroid nodule evaluated?",
        "TSH measurement followed by ultrasound characterization. Fine needle aspiration biopsy for suspicious nodules. Bethesda classification guides management from benign monitoring to surgical excision.",
    ),
    (
        "What is the pathophysiology of heart failure?",
        "Reduced cardiac output triggers neurohormonal activation including RAAS and sympathetic nervous system. This leads to volume overload, ventricular remodeling, myocardial fibrosis, and progressive dysfunction.",
    ),
    (
        "How is chronic hepatitis B managed?",
        "Antiviral therapy with tenofovir or entecavir for active disease. Monitoring includes HBV DNA levels, liver enzymes, and hepatocellular carcinoma screening. Pegylated interferon is an alternative.",
    ),
    (
        "What are the treatment options for breast cancer?",
        "Treatment depends on stage and molecular subtype. Options include surgery, radiation, chemotherapy, hormonal therapy for ER-positive tumors, and targeted therapy such as trastuzumab for HER2-positive disease.",
    ),
    (
        "How is diabetic ketoacidosis treated?",
        "IV fluid resuscitation with normal saline, continuous insulin infusion, potassium replacement, monitoring of anion gap closure, and transition to subcutaneous insulin when eating and gap closes.",
    ),
    (
        "What is the management of acute upper GI bleeding?",
        "Resuscitation with IV fluids and blood products. IV proton pump inhibitor. Urgent endoscopy within 24 hours for diagnosis and treatment. Interventional radiology or surgery for refractory bleeding.",
    ),
    (
        "How is multiple sclerosis diagnosed?",
        "McDonald criteria require dissemination in space and time. MRI shows periventricular, juxtacortical, infratentorial, and spinal cord white matter lesions. CSF oligoclonal bands support diagnosis.",
    ),
    (
        "What is the approach to thyrotoxicosis management?",
        "Beta-blockers for symptom control. Antithyroid drugs methimazole or propylthiouracil. Radioactive iodine ablation. Thyroidectomy for refractory cases, large goiter, or suspected malignancy.",
    ),
    (
        "How is chronic pain managed?",
        "Multimodal approach including physical therapy, cognitive behavioral therapy, non-opioid analgesics, adjuvant medications like gabapentin or duloxetine, and interventional procedures. Opioids as last resort.",
    ),
    (
        "What are the criteria for metabolic syndrome?",
        "Three of five criteria: waist circumference above 102 cm in men or 88 cm in women, triglycerides above 150, HDL below 40 in men or 50 in women, blood pressure above 130/85, fasting glucose above 100.",
    ),
    (
        "How is anaphylaxis treated?",
        "Intramuscular epinephrine 0.3 to 0.5 mg in the anterolateral thigh is the first-line treatment. Adjuncts include IV fluids, antihistamines, corticosteroids, bronchodilators, and airway management.",
    ),
    (
        "What is the workup for a pulmonary nodule?",
        "CT characterization of size, margins, and density. Low-risk nodules monitored with serial imaging. High-risk nodules evaluated with PET-CT and biopsy. Lung-RADS or Fleischner criteria guide management.",
    ),
    (
        "How is Crohn disease treated?",
        "Induction with corticosteroids or biologics like infliximab. Maintenance with thiopurines, methotrexate, or biologics. Surgery for complications. Nutritional support and smoking cessation.",
    ),
    (
        "What are the features of acute respiratory distress syndrome?",
        "ARDS is defined by acute onset, bilateral opacities on imaging, respiratory failure not explained by cardiac failure, and PaO2/FiO2 ratio categorization: mild 200 to 300, moderate 100 to 200, severe below 100.",
    ),
    (
        "How is chronic hepatitis C treated?",
        "Direct-acting antivirals achieve cure rates above 95 percent. Common regimens include sofosbuvir-velpatasvir or glecaprevir-pibrentasvir for 8 to 12 weeks depending on genotype and cirrhosis status.",
    ),
]

TEST_PROMPTS: Final[list[str]] = [
    "Question: What is the recommended initial treatment for a patient presenting with acute myocardial infarction?\nAnswer:",
    "Question: How should a clinician evaluate a patient with suspected pulmonary embolism?\nAnswer:",
    "Question: What are the key diagnostic criteria for systemic lupus erythematosus?\nAnswer:",
    "Question: Describe the management approach for a patient with newly diagnosed type 2 diabetes.\nAnswer:",
    "Question: What is the pathophysiology and treatment of acute pancreatitis?\nAnswer:",
]


# ------------------------------------------------------------------
# Synthetic Dataset
# ------------------------------------------------------------------
class SyntheticMedicalQADataset(Dataset):
    """Synthetic medical QA dataset for TASFT validation training.

    Generates tokenized input_ids and labels from hardcoded QA pairs.
    Each sample is formatted as "Question: ... Answer: ..." and tokenized
    to max_seq_len with padding and truncation.

    Preconditions:
        - tokenizer must have pad_token set
        - max_seq_len > 0

    Postconditions:
        - All returned tensors have shape [max_seq_len]
        - Labels use -100 for padding positions

    Complexity: O(1) per __getitem__ (pre-tokenized).
    """

    def __init__(
        self,
        tokenizer: Any,
        qa_pairs: list[tuple[str, str]],
        num_samples: int,
        max_seq_len: int,
    ) -> None:
        self.samples: list[dict[str, torch.Tensor]] = []
        num_pairs = len(qa_pairs)

        for i in range(num_samples):
            q, a = qa_pairs[i % num_pairs]
            text = f"Question: {q}\nAnswer: {a}"

            encoded = tokenizer(
                text,
                max_length=max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)

            # Labels: copy input_ids, set padding to -100
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            self.samples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.samples[idx]


# ------------------------------------------------------------------
# Helper: SHA-256 file checksum
# ------------------------------------------------------------------
def compute_sha256(filepath: str | Path) -> str:
    """Compute SHA-256 hex digest of a file.

    Args:
        filepath: Path to file.

    Returns:
        64-character lowercase hex SHA-256 digest.

    Complexity: O(file_size), reads in 64KB chunks.
    """
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()


# ------------------------------------------------------------------
# Helper: directory size
# ------------------------------------------------------------------
def dir_size_bytes(path: str | Path) -> int:
    """Compute total size in bytes of all files in a directory tree.

    Args:
        path: Root directory.

    Returns:
        Total bytes.

    Complexity: O(num_files).
    """
    total = 0
    root = Path(path)
    if root.is_file():
        return root.stat().st_size
    for child in root.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


# ------------------------------------------------------------------
# Step 1-2: Load model and tokenizer, create dataset
# ------------------------------------------------------------------
def setup_model_and_data(output_dir: Path) -> tuple[Any, Any, Any]:
    """Download model/tokenizer and create synthetic dataset.

    Returns:
        (model, tokenizer, dataset) tuple.
    """
    log.info("setup_start", model_id=MODEL_ID)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    log.info(
        "model_loaded",
        num_params=sum(p.numel() for p in model.parameters()),
        dtype=str(next(model.parameters()).dtype),
    )

    dataset = SyntheticMedicalQADataset(
        tokenizer=tokenizer,
        qa_pairs=MEDICAL_QA_PAIRS,
        num_samples=NUM_SYNTHETIC_SAMPLES,
        max_seq_len=MAX_SEQ_LEN,
    )
    log.info("dataset_created", num_samples=len(dataset), max_seq_len=MAX_SEQ_LEN)

    return model, tokenizer, dataset


# ------------------------------------------------------------------
# Step 3: Patch model with TASFT gates
# ------------------------------------------------------------------
def patch_with_tasft(model: Any) -> dict[int, Any]:
    """Patch model attention layers with TASFT AttnGate modules.

    Qwen2Attention uses `num_key_value_groups` instead of `num_heads`, so
    we inject the `num_heads` attribute from the model config before calling
    patch_model_attention. This is a workaround -- we do NOT modify TASFT source.

    Returns:
        patched_layers dict mapping layer_idx -> TASFTAttention.
    """
    from tasft.modules.tasft_attention import GateConfig, patch_model_attention

    # Freeze entire model before patching. patch_model_attention freezes
    # attention params, but _verify_frozen_base checks ALL non-gate params.
    # Without this, MLP, layernorm, embeddings etc. remain unfrozen and
    # verification fails.
    for param in model.parameters():
        param.requires_grad = False

    # Defensive: ensure num_heads and num_key_value_heads exist as direct
    # attributes on each attention module. Upstream _extract_attn_dims and
    # _prepare_qkv now handle this, but we inject from config as a fallback
    # for older versions of the TASFT patching code.
    num_attention_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    for layer in model.model.layers:
        attn = layer.self_attn
        if not hasattr(attn, "num_heads"):
            attn.num_heads = num_attention_heads
        if not hasattr(attn, "num_key_value_heads"):
            attn.num_key_value_heads = num_kv_heads

    gate_config = GateConfig(
        block_size=BLOCK_SIZE,
        num_layers=NUM_LAYERS,
        default_threshold=0.5,
    )

    patched_layers = patch_model_attention(model, gate_config)

    log.info(
        "tasft_patch_applied",
        num_patched_layers=len(patched_layers),
        block_size=BLOCK_SIZE,
    )
    return patched_layers


# ------------------------------------------------------------------
# Step 4: Apply LoRA
# ------------------------------------------------------------------
def apply_lora(model: Any) -> Any:
    """Apply LoRA adapters to the model via PEFT.

    Returns:
        PEFT-wrapped model.
    """
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(
        "lora_applied",
        rank=LORA_RANK,
        alpha=LORA_ALPHA,
        trainable_params=trainable,
        total_params=total,
        trainable_pct=round(100.0 * trainable / total, 4),
    )
    return model


# ------------------------------------------------------------------
# Step 5: Train with TASFTTrainer
# ------------------------------------------------------------------
def run_training(
    model: Any,
    tokenizer: Any,
    dataset: Any,
    patched_layers: dict[int, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Run TASFT co-training for NUM_TRAIN_STEPS steps.

    Returns:
        dict with training results including loss curves and timing.
    """
    from transformers import DataCollatorForLanguageModeling

    from tasft.training.trainer import TASFTTrainer, TASFTTrainingArguments

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    training_args = TASFTTrainingArguments(
        output_dir=str(checkpoint_dir),
        max_steps=NUM_TRAIN_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=5,
        max_grad_norm=1.0,
        logging_steps=10,
        save_steps=CHECKPOINT_INTERVAL,
        save_total_limit=2,
        bf16=False,
        fp16=False,
        use_cpu=True,
        dataloader_pin_memory=False,
        report_to="none",
        remove_unused_columns=False,
        # TASFT-specific args
        lambda_gate=LAMBDA_GATE,
        beta_sparse=BETA_SPARSE,
        tau_target=TAU_TARGET,
        gate_lr_ratio=GATE_LR_RATIO,
        gate_warmup_steps=GATE_WARMUP_STEPS,
        layers_per_step=LAYERS_PER_STEP,
        block_size=BLOCK_SIZE,
        rotation_strategy="round_robin",
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = TASFTTrainer(
        model=model,
        args=training_args,
        patched_layers=patched_layers,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    log.info("training_start", max_steps=NUM_TRAIN_STEPS)
    wall_start = time.monotonic()

    train_result = trainer.train()

    wall_time = time.monotonic() - wall_start

    # Extract metrics
    metrics = train_result.metrics
    train_loss = metrics.get("train_loss", float("nan"))

    # Collect per-layer sparsity from the last step's gate outputs
    per_layer_sparsity: list[float] = []
    for idx in sorted(patched_layers.keys()):
        tasft_attn = patched_layers[idx]
        gate_out = getattr(tasft_attn, "_last_gate_output", None)
        if gate_out is not None:
            per_layer_sparsity.append(float(gate_out.sparsity_ratio))
        else:
            per_layer_sparsity.append(0.0)

    mean_sparsity = sum(per_layer_sparsity) / len(per_layer_sparsity) if per_layer_sparsity else 0.0

    # Extract gate loss from trainer log history. The TASFTTrainer logs
    # loss_gate per logging step; take the last logged value.
    gate_loss_final = 0.0
    log_history = getattr(trainer, "state", None)
    if log_history is not None:
        for entry in reversed(getattr(log_history, "log_history", [])):
            if "loss_gate" in entry:
                gate_loss_final = float(entry["loss_gate"])
                break

    # Estimate tokens/second during training
    total_tokens = NUM_TRAIN_STEPS * BATCH_SIZE * MAX_SEQ_LEN
    tokens_per_second = total_tokens / wall_time if wall_time > 0 else 0.0

    result = {
        "steps": NUM_TRAIN_STEPS,
        "final_loss": float(train_loss),
        "loss_task": float(train_loss - gate_loss_final),
        "loss_gate": gate_loss_final,
        "mean_sparsity": mean_sparsity,
        "per_layer_sparsity": per_layer_sparsity,
        "wall_time_seconds": round(wall_time, 2),
        "tokens_per_second": round(tokens_per_second, 2),
    }

    log.info(
        "training_complete",
        final_loss=round(train_loss, 6),
        mean_sparsity=round(mean_sparsity, 4),
        wall_time_s=round(wall_time, 2),
        tokens_per_s=round(tokens_per_second, 2),
    )

    return result


# ------------------------------------------------------------------
# Step 6: Export deployment bundle
# ------------------------------------------------------------------
def export_bundle(
    model: Any,
    patched_layers: dict[int, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Export gate weights, kernel config, and manifest as a deployment bundle.

    Bundle structure:
        bundle/
            gates/layer_{i}_gate.safetensors  (per-layer gate weights)
            kernel_config.json                 (per-layer thresholds)
            manifest.json                      (SHA-256 checksums, metadata)

    Returns:
        dict with bundle metadata for the report.
    """
    bundle_dir = output_dir / "bundle"
    gates_dir = bundle_dir / "gates"
    gates_dir.mkdir(parents=True, exist_ok=True)

    log.info("bundle_export_start", bundle_dir=str(bundle_dir))

    checksums: dict[str, str] = {}
    per_layer_configs: dict[str, Any] = {}

    # Save gate weights as SafeTensors, one file per layer
    for idx in sorted(patched_layers.keys()):
        tasft_attn = patched_layers[idx]
        gate = tasft_attn.gate
        gate_state = gate.state_dict()

        gate_filename = f"layer_{idx}_gate.safetensors"
        gate_path = gates_dir / gate_filename
        save_file(gate_state, str(gate_path))

        relative_path = f"gates/{gate_filename}"
        checksums[relative_path] = compute_sha256(gate_path)

        # Get sparsity from last gate output if available
        gate_out = getattr(tasft_attn, "_last_gate_output", None)
        achieved_sparsity = float(gate_out.sparsity_ratio) if gate_out is not None else 0.0

        # Gate loss for this layer comes from the last gate output if available.
        # After warmup (step 10+), the gate loss reflects actual KL divergence.
        gate_loss_val = float(gate_out.gate_loss) if gate_out is not None and hasattr(gate_out, "gate_loss") else 0.0

        per_layer_configs[str(idx)] = {
            "layer_idx": idx,
            "threshold_tau": float(gate.default_threshold),
            "target_sparsity": TAU_TARGET,
            "achieved_sparsity_validation": achieved_sparsity,
            "gate_loss_validation": gate_loss_val,
            "block_size": BLOCK_SIZE,
        }

    # Save kernel_config.json
    kernel_config = {
        "block_size": BLOCK_SIZE,
        "global_threshold": 0.5,
        "per_layer_config": {
            int(k): v for k, v in per_layer_configs.items()
        },
        "min_sparsity_for_speedup": 0.5,
    }
    kernel_config_path = bundle_dir / "kernel_config.json"
    with open(kernel_config_path, "w") as f:
        json.dump(kernel_config, f, indent=2)
    checksums["kernel_config.json"] = compute_sha256(kernel_config_path)

    # Save manifest.json
    manifest = {
        "version": "1.0.0",
        "bundle_format_version": "1.0",
        "model_name": "tasft-qwen2.5-0.5b-medical-validation",
        "base_model_id": MODEL_ID,
        "domain": "medical_qa",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_hash": "validation_run",
        "training_args_hash": compute_sha256_string(json.dumps({
            "lr": LEARNING_RATE,
            "steps": NUM_TRAIN_STEPS,
            "lambda_gate": LAMBDA_GATE,
            "beta_sparse": BETA_SPARSE,
        })),
        "checksums": checksums,
        "total_size_bytes": dir_size_bytes(bundle_dir),
        "num_layers": NUM_LAYERS,
    }
    manifest_path = bundle_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Verify checksums
    checksum_verified = True
    for rel_path, expected_hash in checksums.items():
        actual_hash = compute_sha256(bundle_dir / rel_path)
        if actual_hash != expected_hash:
            checksum_verified = False
            log.error("checksum_mismatch", file=rel_path, expected=expected_hash, actual=actual_hash)

    total_size = dir_size_bytes(bundle_dir)
    num_gate_files = len([k for k in checksums if k.startswith("gates/")])

    log.info(
        "bundle_export_complete",
        bundle_dir=str(bundle_dir),
        num_gate_files=num_gate_files,
        total_size_bytes=total_size,
        checksum_verified=checksum_verified,
    )

    return {
        "path": str(bundle_dir),
        "num_gate_files": num_gate_files,
        "total_size_bytes": total_size,
        "checksum_verified": checksum_verified,
    }


def compute_sha256_string(s: str) -> str:
    """Compute SHA-256 hex digest of a string.

    Args:
        s: Input string.

    Returns:
        64-character lowercase hex digest.
    """
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------
# Step 7: Inference
# ------------------------------------------------------------------
def run_inference(
    model: Any,
    tokenizer: Any,
) -> dict[str, Any]:
    """Run inference on test prompts and measure throughput.

    Returns:
        dict with inference results for the report.
    """
    log.info("inference_start", num_prompts=NUM_INFERENCE_PROMPTS)

    model.eval()
    completions: list[str] = []
    total_new_tokens = 0
    total_gen_time = 0.0

    for i, prompt in enumerate(TEST_PROMPTS):
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=MAX_SEQ_LEN,
            truncation=True,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        prompt_len = input_ids.shape[1]

        gen_start = time.monotonic()
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        gen_time = time.monotonic() - gen_start

        new_token_ids = output_ids[0][prompt_len:]
        new_tokens = len(new_token_ids)
        total_new_tokens += new_tokens
        total_gen_time += gen_time

        completion = tokenizer.decode(new_token_ids, skip_special_tokens=True)
        completions.append(completion)

        log.info(
            "inference_prompt_done",
            prompt_idx=i,
            new_tokens=new_tokens,
            gen_time_s=round(gen_time, 3),
            tokens_per_s=round(new_tokens / gen_time, 2) if gen_time > 0 else 0.0,
        )

    mean_tps = total_new_tokens / total_gen_time if total_gen_time > 0 else 0.0

    log.info(
        "inference_complete",
        total_new_tokens=total_new_tokens,
        total_gen_time_s=round(total_gen_time, 3),
        mean_tokens_per_s=round(mean_tps, 2),
    )

    return {
        "num_prompts": NUM_INFERENCE_PROMPTS,
        "mean_tokens_per_second": round(mean_tps, 2),
        "completions": completions,
    }


# ------------------------------------------------------------------
# Step 8: Generate report
# ------------------------------------------------------------------
def generate_report(
    training_results: dict[str, Any],
    bundle_results: dict[str, Any],
    inference_results: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Generate and save the validation report JSON.

    Returns:
        The report dict.
    """
    per_layer = training_results.get("per_layer_sparsity", [])
    layers_above_target = sum(1 for s in per_layer if s >= TAU_TARGET)

    report = {
        "model": MODEL_ID,
        "hardware": {
            "cpu": "Apple M4 Pro",
            "ram_gb": 48,
            "gpu": "none",
        },
        "training": {
            "steps": training_results["steps"],
            "final_loss": training_results["final_loss"],
            "loss_task": training_results["loss_task"],
            "loss_gate": training_results["loss_gate"],
            "mean_sparsity": training_results["mean_sparsity"],
            "wall_time_seconds": training_results["wall_time_seconds"],
            "tokens_per_second": training_results["tokens_per_second"],
        },
        "bundle": bundle_results,
        "inference": inference_results,
        "gate_analysis": {
            "per_layer_sparsity": per_layer,
            "mean_sparsity": training_results["mean_sparsity"],
            "layers_above_target": layers_above_target,
        },
    }

    report_path = output_dir / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    log.info("report_saved", path=str(report_path))
    return report


# ------------------------------------------------------------------
# Step 9: Print summary
# ------------------------------------------------------------------
def print_summary(report: dict[str, Any]) -> None:
    """Print a human-readable summary table to stdout."""
    print("\n" + "=" * 72)
    print("  TASFT End-to-End Validation Summary")
    print("=" * 72)

    t = report["training"]
    print(f"\n  Model:               {report['model']}")
    print(f"  Hardware:            {report['hardware']['cpu']} / {report['hardware']['ram_gb']}GB RAM / {report['hardware']['gpu']}")

    print(f"\n  Training Steps:      {t['steps']}")
    print(f"  Final Loss:          {t['final_loss']:.6f}")
    print(f"  Mean Sparsity:       {t['mean_sparsity']:.4f}")
    print(f"  Wall Time:           {t['wall_time_seconds']:.1f}s")
    print(f"  Tokens/sec:          {t['tokens_per_second']:.1f}")

    b = report["bundle"]
    print(f"\n  Bundle Path:         {b['path']}")
    print(f"  Gate Files:          {b['num_gate_files']}")
    print(f"  Total Size:          {b['total_size_bytes'] / 1024:.1f} KB")
    print(f"  Checksum Verified:   {b['checksum_verified']}")

    inf = report["inference"]
    print(f"\n  Inference Prompts:   {inf['num_prompts']}")
    print(f"  Mean Tokens/sec:     {inf['mean_tokens_per_second']:.1f}")

    g = report["gate_analysis"]
    print(f"\n  Layers Above Target: {g['layers_above_target']}/{len(g['per_layer_sparsity'])}")
    print(f"  Mean Gate Sparsity:  {g['mean_sparsity']:.4f}")

    print("\n  Completions (first 100 chars):")
    for i, comp in enumerate(inf["completions"]):
        truncated = comp[:100].replace("\n", " ")
        print(f"    [{i}] {truncated}...")

    print("\n" + "=" * 72)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main() -> int:
    """Run the full E2E validation pipeline.

    Returns:
        0 on success, 1 on failure.
    """
    parser = argparse.ArgumentParser(description="TASFT E2E Validation")
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for all outputs (checkpoints, bundle, report)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Deterministic seeds
    torch.manual_seed(SEED)

    training_results: dict[str, Any] = {}
    bundle_results: dict[str, Any] = {}
    inference_results: dict[str, Any] = {}

    try:
        # Step 1-2: Model + Data
        model, tokenizer, dataset = setup_model_and_data(output_dir)

        # Step 3: Patch with TASFT
        patched_layers = patch_with_tasft(model)

        # Step 4: Apply LoRA
        model = apply_lora(model)

        # Re-extract patched layers after PEFT wrapping: PEFT wraps the model,
        # but the TASFTAttention references in patched_layers still point to the
        # actual attention modules inside the wrapped model. Verify they are
        # still accessible.
        log.info("verifying_patched_layers_post_lora")
        for idx, tasft_attn in patched_layers.items():
            gate_params = sum(p.numel() for p in tasft_attn.gate.parameters())
            if gate_params == 0:
                log.warning("gate_no_params", layer=idx)

        # Step 5: Train
        training_results = run_training(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            patched_layers=patched_layers,
            output_dir=output_dir,
        )

        # Step 6: Export bundle
        bundle_results = export_bundle(
            model=model,
            patched_layers=patched_layers,
            output_dir=output_dir,
        )

        # Step 7: Inference
        # Re-enable use_cache for generation
        if hasattr(model, "config"):
            model.config.use_cache = True
        elif hasattr(model, "base_model") and hasattr(model.base_model, "config"):
            model.base_model.config.use_cache = True

        inference_results = run_inference(model=model, tokenizer=tokenizer)

    except Exception:
        log.error("pipeline_error", exc_info=True)
        traceback.print_exc()

        # Fill in empty results for report
        if not training_results:
            training_results = {
                "steps": 0, "final_loss": float("nan"), "loss_task": float("nan"),
                "loss_gate": 0.0, "mean_sparsity": 0.0, "per_layer_sparsity": [],
                "wall_time_seconds": 0.0, "tokens_per_second": 0.0,
            }
        if not bundle_results:
            bundle_results = {
                "path": "", "num_gate_files": 0,
                "total_size_bytes": 0, "checksum_verified": False,
            }
        if not inference_results:
            inference_results = {
                "num_prompts": 0, "mean_tokens_per_second": 0.0,
                "completions": [],
            }

    # Step 8: Generate report
    report = generate_report(
        training_results=training_results,
        bundle_results=bundle_results,
        inference_results=inference_results,
        output_dir=output_dir,
    )

    # Step 9: Print summary
    print_summary(report)

    # Return status
    success = (
        training_results.get("steps", 0) == NUM_TRAIN_STEPS
        and bundle_results.get("checksum_verified", False)
        and len(inference_results.get("completions", [])) == NUM_INFERENCE_PROMPTS
    )

    if success:
        log.info("validation_passed")
        return 0
    log.warning("validation_incomplete")
    return 1


if __name__ == "__main__":
    sys.exit(main())
