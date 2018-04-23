import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

from FeatureSelector import FeatureSelector
from preproccess import read_data, prepare_and_scale_data

train, test = read_data()
train, y_train, test, id_test = prepare_and_scale_data(train, test)

feature_selector = FeatureSelector(train,test)
predictors = feature_selector.feature_selection_based_on_genetic_algo(train,test,y_train)

train = train[predictors]


en = LinearRegression(fit_intercept=True, n_jobs=-1)

rf = RandomForestRegressor(n_estimators=100, n_jobs=2, max_depth=6,)

et = ExtraTreesRegressor(n_estimators=100, n_jobs=4, max_depth=6,)

xgbm = xgb.sklearn.XGBRegressor(max_depth=6, learning_rate=0.05,
                                n_estimators=1000, base_score=y_train.mean())

lgbm = lgb.LGBMRegressor(nthread=3,silent=True,learning_rate=0.05,max_depth=7,n_estimators=1000)


results = cross_val_score(xgbm, train, y_train, cv=5, scoring='r2')
print("XCBOOST score: %.4f (%.4f)" % (results.mean(), results.std()))

results = cross_val_score(lgbm, train, y_train, cv=5, scoring='r2')
print("LGBOOST score: %.4f (%.4f)" % (results.mean(), results.std()))

results = cross_val_score(en, train, y_train, cv=5, scoring='r2')
print("LogisticRegression score: %.4f (%.4f)" % (results.mean(), results.std()))

results = cross_val_score(rf, train, y_train, cv=5, scoring='r2')
print("RandomForest score: %.4f (%.4f)" % (results.mean(), results.std()))

results = cross_val_score(et, train, y_train, cv=5, scoring='r2')
print("ExtraTrees score: %.4f (%.4f)" % (results.mean(), results.std()))
