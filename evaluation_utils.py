"""
Módulo de utilidades para evaluación de modelos de detección de fraude en SMS.
Este archivo exporta funciones puras sin dependencias de rutas ni datos.
"""
import matplotlib.pyplot as plt  # Para gráficos
import seaborn as sns  # Para graficar matrices de confusión
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_auc_score,
    RocCurveDisplay
)

# ==============================
# FUNCIONES DE EVALUACIÓN
# ==============================
def evaluate_model(y_true, y_pred, y_prob=None):
    """
    Imprime métricas clave de clasificación:
     - Reporte completo (precision, recall, f1 por clase)
     - Accuracy y F1 macro
     - AUC ROC (si se pasa y_prob)
    """
    from sklearn.metrics import (
        classification_report,
        accuracy_score,
        f1_score,
        roc_auc_score
    )

    print("\n📊 Reporte de Clasificación:")
    print(classification_report(y_true, y_pred, zero_division=0))

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")

    if y_prob is not None:
        auc = roc_auc_score(y_true, y_prob)
        print(f"AUC ROC: {auc:.4f}")



def plot_confusion_matrix(y_true, y_pred, labels=None):
    """
    Grafica la matriz de confusión.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_true, y_prob):
    """
    Grafica la curva ROC y muestra AUC si el modelo da probabilidades.
    y_prob debe ser la probabilidad de la clase positiva.
    """
    if y_prob is None:
        print("\n⚠️ El modelo no soporta predict_proba para curva ROC.")
        return
    auc = roc_auc_score(y_true, y_prob)
    print(f"\nAUC ROC Score: {auc:.4f}")
    RocCurveDisplay.from_predictions(y_true, y_prob)
    plt.title("Curva ROC")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


