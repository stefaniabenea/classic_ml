from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


models = {"Support Vector Machine":SVC(kernel="rbf", C=1, gamma='scale'),
            "Logistic Regression":LogisticRegression(max_iter=1000),
            "k-Nearest Neighbours":KNeighborsClassifier(n_neighbors=5),
            "Decision Tree":DecisionTreeClassifier(max_depth=5,random_state=42),
            "Random Forest":RandomForestClassifier(n_estimators=100, random_state=42)
            }