"""Train and evaluate random forest on the shared dataset."""
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from pathsnap.config import CFG
from models.logreg.training import run

def main():
    run('random_forest','ensemble (bagging)',lambda:make_pipeline(RandomForestClassifier(n_estimators=20,max_depth=15,min_samples_leaf=5,n_jobs=2,random_state=CFG['seed'])))
if __name__=='__main__': main()
