# ♟️ Chess Move Prediction Model (Overview)

## Model Architecture

This model is a custom chess move prediction policy network designed to output the best move for a given board position. It uses a combination of convolutional layers, residual blocks, and a legality mask to ensure only legal moves are predicted.

### **Input Representation (18 Planes)**

* 12 planes for piece placements (6 white, 6 black)
* 1 plane for side to move
* 4 planes for castling rights
* 1 plane for en-passant target square

Shape: **8 × 8 × 18**

---

## **Neural Network Architecture**

### **1. Initial Convolution Layer**

* Conv2D (128 filters, 3×3)
* BatchNorm
* ReLU

### **2. Residual Blocks (×3)**

Each block contains:

* Conv2D → BN → ReLU
* Conv2D → BN
* Squeeze-and-Excitation (SE) attention
* Skip connection + ReLU

### **3. Policy Head**

* GlobalAveragePooling2D
* Dense (2048 units, ReLU)
* Dropout
* Dense (1629 units) → raw logits

### **4. Legality Mask (MaskedAdd Layer)**

The model adds `log(mask)` to logits, zeroing out illegal moves.

---

## **Loss Function**

* **Masked Sparse Categorical Crossentropy** with Label Smoothing (0.05)

---

## **Optimizer**

* Adam optimizer
* CosineDecayRestarts Schedule
* Gradient clipping

---

## **Training Pipeline**

### Steps:

1. Load base dataset from NPZ
2. Parse additional PGN files
3. Generate FEN-based legality masks
4. Prepare `tf.data` pipelines
5. Train ResNet policy network
6. Save:

   * Checkpoints
   * Final model
   * Accuracy/Loss plots
   * TensorBoard logs

---

## **Inference Pipeline**

1. Convert chess.Board → 18-plane tensor
2. Build legality mask
3. Model forward pass: `model([tensor, mask])`
4. Softmax → masked probabilities
5. Pick top legal move

Supports top-k move analysis.

---

## **Playing Against the Model**

Run:

```
python src/chess_play.py
```

Model:

* Accepts user UCI input (e.g., `e2e4`)
* Predicts best move
* Displays top-3 moves with probabilities

---

## **Files Related to the Model**

* `src/train_base.py` → Main training script
* `src/chess_play.py` → Play against model
* `artifacts/ckpts/` → Saved models
* `artifacts/vocab_improved.json` → Move vocabulary
* `artifacts/dataset_improved.npz` → Encoded training data
#
