from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

accuracies = {}
def evaluate_model(model,X_train_scaled,y_train,X_test_scaled,y_test, name):
    model.fit(X_train_scaled,y_train)
    y_pred= model.predict(X_test_scaled)
    cm = confusion_matrix(y_test,y_pred)
    acc = accuracy_score(y_test,y_pred)*100
    accuracies[name]=acc

    print(f"\n Model: {name}")
    print(f"Accuracy:{acc:.2f}")
    print("Classification report:",classification_report(y_test,y_pred))
    
    disp=ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title(f"Confusion matrix -{name}")
    plt.show()
    