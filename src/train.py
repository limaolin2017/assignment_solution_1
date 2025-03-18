# src/train.py
import wandb
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def train():

    wandb.init(project="my_sweep_project")  

    
    class_weight = wandb.config.class_weight  
    C = wandb.config.C                        
    max_iter = wandb.config.max_iter          

    
    df_msg = pd.read_csv("src/data/available_conversations.csv")
    df_topics = pd.read_csv("src/data/available_topics.csv")
    topic_map = dict(zip(df_topics["topic_id"], df_topics["topic_name"]))
    df_msg["topic"] = df_msg["topic_id"].map(topic_map)

    
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df_msg["message"], df_msg["topic"], test_size=0.2, random_state=42
    )

    
    vec = TfidfVectorizer(max_features=5000)
    X_train = vec.fit_transform(train_texts)
    X_test = vec.transform(test_texts)

    
    model = LogisticRegression(class_weight=class_weight, C=C, max_iter=max_iter)
    model.fit(X_train, train_labels)

    
    y_pred = model.predict(X_test)
    acc = accuracy_score(test_labels, y_pred)
    report = classification_report(test_labels, y_pred, output_dict=True)

    
    wandb.log({
        "accuracy": acc,
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1_score": report["weighted avg"]["f1-score"]
    })

    wandb.finish()

if __name__ == "__main__":
    train()
