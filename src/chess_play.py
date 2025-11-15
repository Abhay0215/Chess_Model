import tensorflow as tf
import numpy as np
import chess
import json
import random
from pathlib import Path
from tensorflow.keras import layers
from tensorflow.keras.utils import register_keras_serializable

# --------------------------------------------------------------
# 🧩 Custom Layers (required)
# --------------------------------------------------------------
@register_keras_serializable()
class MaskedAdd(layers.Layer):
    def call(self, inputs):
        logits, mask = inputs
        return logits + tf.math.log(mask + 1e-9)

@register_keras_serializable()
class MaskedSCE(tf.keras.losses.Loss):
    def __init__(self, smooth=0.05, name="masked_sce", **kwargs):
        super().__init__(name=name, **kwargs)
        self.smooth = smooth

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        C = tf.shape(y_pred)[-1]
        y = tf.one_hot(y_true, C, dtype=tf.float32)
        y = (1.0 - self.smooth) * y + self.smooth / tf.cast(C, tf.float32)
        return tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(y, y_pred, from_logits=True)
        )

# --------------------------------------------------------------
# ⚙️ Paths
# --------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
ART_DIR = BASE_DIR / "artifacts"
VOCAB_PATH = ART_DIR / "vocab_improved.json"
MODEL_PATH = ART_DIR / "ckpts" / "final_trained_model.keras"

# --------------------------------------------------------------
# 📜 Load Correct Vocab
# --------------------------------------------------------------
with open(VOCAB_PATH, "r") as f:
    raw_vocab = json.load(f)

# Your vocab structure is: {"move_to_id": {...}}
if "move_to_id" in raw_vocab:
    mv_map = raw_vocab["move_to_id"]
else:
    mv_map = raw_vocab

move_to_id = {mv: int(idx) for mv, idx in mv_map.items()}
id_to_move = {int(idx): mv for mv, idx in mv_map.items()}

NUM_CLASSES = len(move_to_id)
print(f"✔ Loaded {NUM_CLASSES} moves from vocab_improved.json")

# --------------------------------------------------------------
# 🧠 Load Model
# --------------------------------------------------------------
print(f"Loading model: {MODEL_PATH}")

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"MaskedAdd": MaskedAdd, "MaskedSCE": MaskedSCE},
    compile=False
)

print("✔ Model loaded successfully\n")

# --------------------------------------------------------------
# ♟ Board → 18-plane tensor
# --------------------------------------------------------------
def board_to_tensor(board: chess.Board):
    planes = np.zeros((8, 8, 18), dtype=np.float32)

    for sq, piece in board.piece_map().items():
        plane = (piece.piece_type - 1) + (6 if piece.color == chess.BLACK else 0)
        row, col = divmod(sq, 8)
        planes[7 - row, col, plane] = 1.0

    # game-state channels
    planes[:, :, 12] = 1.0 if board.turn == chess.WHITE else 0.0
    if board.has_kingside_castling_rights(chess.WHITE): planes[:, :, 13] = 1
    if board.has_queenside_castling_rights(chess.WHITE): planes[:, :, 14] = 1
    if board.has_kingside_castling_rights(chess.BLACK): planes[:, :, 15] = 1
    if board.has_queenside_castling_rights(chess.BLACK): planes[:, :, 16] = 1
    if board.ep_square is not None:
        r, c = divmod(board.ep_square, 8)
        planes[7 - r, c, 17] = 1

    return planes

# --------------------------------------------------------------
# ♟ Legality Mask
# --------------------------------------------------------------
def build_mask(board):
    mask = np.zeros((NUM_CLASSES,), dtype=np.float32)
    for mv in board.legal_moves:
        idx = move_to_id.get(mv.uci(), None)
        if isinstance(idx, int):
            mask[idx] = 1.0
    if mask.sum() == 0:
        mask[:] = 1.0
    return mask

# --------------------------------------------------------------
# 🔮 Predict Best Move
# --------------------------------------------------------------
def predict_best_move(board: chess.Board, top_k=3):
    x = board_to_tensor(board)[None, ...]
    m = build_mask(board)[None, ...]

    logits = model.predict([x, m], verbose=0)
    probs = tf.nn.softmax(logits)[0].numpy()
    probs *= m[0]
    probs /= probs.sum() + 1e-12

    top_ids = np.argsort(probs)[::-1][:top_k]
    top_moves = [(id_to_move[i], float(probs[i])) for i in top_ids]

    # choose best legal move
    for mv, p in top_moves:
        if chess.Move.from_uci(mv) in board.legal_moves:
            return mv, p, top_moves

    # fallback: random legal move
    mv = random.choice(list(board.legal_moves)).uci()
    return mv, 0.0, [(mv, 1.0)]

# --------------------------------------------------------------
# 🎮 Game Loop
# --------------------------------------------------------------
def play_vs_model():
    board = chess.Board()
    print("\n♟ New Game Started!\n")

    side = input("Choose your side (w/b): ").strip().lower()

    while not board.is_game_over():
        print("\n------------------------")
        print(board)
        print("------------------------\n")

        # Human turn
        if (board.turn == chess.WHITE and side == "w") or (board.turn == chess.BLACK and side == "b"):
            move_str = input("Your move (e2e4): ").strip()
            try:
                move = chess.Move.from_uci(move_str)
                if move not in board.legal_moves:
                    print("❌ Illegal move.")
                    continue
                board.push(move)
            except:
                print("❌ Invalid move format.")
                continue

        # Model turn
        else:
            print("\n🤖 Model thinking...\n")
            mv, p, tops = predict_best_move(board)
            board.push(chess.Move.from_uci(mv))
            print(f"🤖 Model plays: {mv}  (p={p:.3f})")

            print("\nTop Predictions:")
            for t_mv, t_p in tops:
                print(f"  {t_mv}: {t_p:.3f}")

    print("\n🏁 GAME OVER!")
    print("Result:", board.result())

# --------------------------------------------------------------
# ▶️ RUN
# --------------------------------------------------------------
if __name__ == "__main__":
    play_vs_model()
