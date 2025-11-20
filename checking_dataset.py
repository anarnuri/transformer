from dataset import BarLinkageDataset
import torch

data_dir = "/home/anurizada/Documents/processed_dataset_17"   # <-- your path here

ds = BarLinkageDataset(data_dir)
print(f"Dataset length = {len(ds)}\n")

NUM_EXAMPLES_TO_PRINT = 3

for i in range(NUM_EXAMPLES_TO_PRINT):
    print("="*80)
    print(f"🟦 SAMPLE {i}")
    print("="*80)

    s = ds[i]

    # -----------------------------
    # PRINT SHAPES
    # -----------------------------
    print("images:", s["images"].shape)
    print("decoder_input_discrete:", s["decoder_input_discrete"].shape)
    print("labels_discrete:", s["labels_discrete"].shape)
    print("attention_mask:", s["attention_mask"].shape)
    print("causal_mask:", s["causal_mask"].shape)
    print("encoded_labels:", s["encoded_labels"].shape)

    if "vae_mu" in s:
        print("vae_mu:", s["vae_mu"].shape)

    print("\n")

    # -----------------------------
    # PRINT ACTUAL CONTENT
    # -----------------------------
    print("encoded_labels (mech type):")
    print(s["encoded_labels"].tolist())
    print()

    print("decoder_input_discrete:")
    print(s["decoder_input_discrete"].tolist())
    print()

    print("labels_discrete:")
    print(s["labels_discrete"].tolist())
    print()

    # print("attention_mask:")
    # print(s["attention_mask"].int().tolist())
    # print()

    # print("causal_mask:")
    # print(s["causal_mask"].int().tolist())
    # print()

    # if "vae_mu" in s:
    #     print("vae_mu:")
    #     print(s["vae_mu"].squeeze().tolist())
    #     print()

    print("\n\n")
