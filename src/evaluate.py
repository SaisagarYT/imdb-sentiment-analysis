import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return acc, prec, rec, f1

def plot_confusion_matrix(y_true, y_pred, save_path="reports/figures/confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print("Confusion matrix saved to", save_path)

def analyze_misclassifications(texts, y_true, y_pred, n_samples=3):
    count = 0
    print("\nMisclassified Reviews:")
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            actual = "Positive" if y_true[i] == 1 else "Negative"
            predicted = "Positive" if y_pred[i] == 1 else "Negative"
            print(f"Sample {count + 1}: Actual = {actual}, Predicted = {predicted}")
            print(f"Review: {texts[i][:150]}...\n")
            count += 1
            if count >= n_samples:
                break

if __name__ == "__main__":
    y_true_dummy = [1, 0, 1, 1, 0, 0, 1, 0]
    y_pred_dummy = [1, 0, 1, 0, 0, 1, 1, 0]
    dummy_texts = [f"Sample review text number {i}" for i in range(8)]
    
    acc, prec, rec, f1 = compute_metrics(y_true_dummy, y_pred_dummy)
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    
    plot_confusion_matrix(y_true_dummy, y_pred_dummy)
    analyze_misclassifications(dummy_texts, y_true_dummy, y_pred_dummy, n_samples=2)
