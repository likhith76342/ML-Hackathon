"""Train and evaluate logistic regression on the shared dataset."""
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from pathsnap.config import CFG
from .training import run

def main():
    run('logreg','linear',lambda:make_pipeline(StandardScaler(),LogisticRegression(max_iter=500,random_state=CFG['seed'])))
if __name__=='__main__': main()
