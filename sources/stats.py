# Pour le multiprocessing
import concurrent.futures
from multiprocessing import cpu_count

from statsmodels.api import stats, tsa
from pandas import concat, DataFrame, Series, PeriodIndex
from numpy import array, concatenate, square, sum, mean, sqrt

# Pour mesurer la performance temporelle
from time import perf_counter

def df_construct_predic_ci(index, pred, ci):
	"""
	__Description__:
	  ...
	__Parametres__:
	  ...
	__Return__:
	  ...
	"""
	df_pred_ci = concat([pred, ci], axis=1)
	df_pred_ci.index = index
	return df_pred_ci

 

def get_prediction_values(resultmodel, mode='raw'):
	"""
	__Description__:
	  La fonction permet de générer les prédictions du modèle ajusté au
	  préalable sur un intervalle déréférencé de la même longueur que
	  le set d'entraînement sur lequel le modèle a été ajusté.
	  Le paramètre "mode" permet de choisir si nous prenons toutes les
	  valeurs du training set sans filtrage ou si nous
	__Parametres__:
	  ...
	__Return__:
	  ...
	"""
	l_train = resultmodel.nobs
	if mode == "raw":
		predict_values = resultmodel.predict(0, l_train - 1, alpha=0.1)        
	return (predict_values)

 

def get_prediction_conf_int(resultmodel, ci=0.9):
	"""
	__Description__:
	  ...
	__Parametres__:
	  ...
	__Return__:
	  ...
	"""
	l_train = resultmodel.nobs
	conf_int = resultmodel.get_prediction(0, l_train - 1, alpha=1 - ci).conf_int()
	return (conf_int)

 

def apply_mdl_newdata_Pred_Test_Fcst(resultmodel, s_data, s_forecast, ratio_traint_test=0.8):
	"""
	__Description__:
	  ...
	__Parametres__:
	  ...
	__Return__:
	  ...
	"""
	l_train = resultmodel.nobs
	l_test = s_data.shape[0] - l_train
	df_data = s_data.to_frame()
	df_forecast = s_forecast.to_frame()
	dict_model = {}
	dict_model['modele'] = resultmodel.model
	dict_model['resultats'] = resultmodel
	
	predict = get_prediction_values(resultmodel)
	ci_predict = get_prediction_conf_int(resultmodel)
	df_predict = concat([predict, ci_predict], axis=1)
	df_predict.index = s_data.index[:l_train]
	dict_model['predict'] = df_predict
	
	dict_model['predict_test'] = forecast_nth_step_forward(resultmodel, df_data.iloc[-l_test:].reset_index(), 'Nom de la colonne')
	dict_model['forecast'] = forecast_nth_step_forward(resultmodel, df_forecast.reset_index(), Nom de la colonne')
	return(dict_model)

 

def test_model(model, test_serie):
	"""
	__Description__:
	  ...
	__Parametres__:
	  ...
	__Return__:
	  ...
	"""
	pass

 

def forecast_n_step_forward(model_result, df_to_fcst, column_name, ci=0.9, fcst_window=1, refit_model=False):
	"""
	__Description__:
	  Fonction permettant de faire de la prevision a l'aide du modele passé en parametre
	  (en realite on passe l'objet SARIMAXResult).
	  Il faut egalement donner le dataframe et le nom de la colonne qu'il faut predire.
	  Le modele permet de predire fcst_window date dans le futur à partir de la dernière
	  date à laquelle le modèle a été ajusté.
	  La fonction construit l'intervalle de prédiction par tranche de fcst_window valeurs
	  en ajoutant progressivement les valeurs de la serie temporelle non encore vues par
	  le modèle.
	__Parametres__:
	  ...
	__Return__:
	  ...
	__Remarks__:
	  ATTENTION, il faut reset l'index du dataframe avant / donner dataframe.reset_index.
	  A cause de la frequence de l'index, statsmodels peut generer une mauvaise
	  date en faisant le forecast (il doit prendre la premiere valeur et regarder cb)
	  d'elements de ligne il y a dans le df, mais du coup il se fait avoir car il ne
	  prend pas en compte les jours feries ...
	"""
	forecast_values = array([[0.42, 0.42, 0.42]])
	l_to_fcst = df_to_fcst.shape[0]
	cpy_model_result = model_result
	
	if isinstance(df_to_fcst, DataFrame):
		serie = df_to_fcst[column_name]
	if isinstance(df_to_fcst, Series):
		serie = df_to_fcst
	dummy = []
	for i in range(0, l_to_fcst + 1, fcst_window):
		if (i + fcst_window  > l_to_fcst):
			break
		date = serie.index[i]
		ret=cpy_model_result.get_prediction(start=date, end=date + fcst_window - 1, dynamic=False, alpha=1-ci)
		ret = ret.summary_frame(alpha=1-ci)
		dummy.append(ret.index)
		ret=ret[['mean', 'mean_ci_lower', 'mean_ci_upper']]
		forecast_values = concatenate((forecast_values, ret.values.reshape(-1, 3)))        
	
		tmp = array([serie.loc[date:date+fcst_window]]).reshape(-1,1)
		cpy_model_result=cpy_model_result.append(endog=tmp, refit=refit_model)
	
	forecast_values=DataFrame({'DateTraitement':df_to_fcst.index[0:i],
							   'fcst':forecast_values[1:i+1,0].reshape(-1,),
							   'ci_lower':forecast_values[1:i+1,1].reshape(-1,),
							   'ci_upper':forecast_values[1:i+1,2].reshape(-1,)}).set_index('DateTraitement')
	return (forecast_values)

 

def batch_forecast_n_step_forward(selected_models, df_to_fcst, column_name, **kwargs):
	"""
	__Description__:
	  Permet de determiner l'intervalle de prédiction via la fonction forecast_n_step_forward
	  pour un ensemble de modèles.
	  
	__Parametres__:
	  selected_models: [list] de l'ensemble de modèles pour lesquels les intervalles
		  de prédiction seront déterminés.
	  df_to_fcst     : [DataFrame] correspondant à la portion de valeurs non utilisées pour
		  l'ajustement et qui sera progressivement ajouté aux modèles afin de construire
		  les intervalles de prédiction
	  column_name    : [str] nom de la colonne du DataFrame devant être prédite.
	  **kwargs       : [dict] paramètres pour la fonction forecast_n_step_forward
		  pouvant contenir: ci, fcst_window, refit_model
		  
	__Return__:
	  Rien, la fonction ajoute directement une entrée 'forecast' dans les dictionnaires
	  constituant les différents modèles.
	"""
	ci=kwargs.get('ci', 0.9)
	f_w=kwargs.get('fcst_window', 1)
	refit=kwargs.get('refit_model', False)
	n_cpu = cpu_count()
	
	t_start = perf_counter()
	forecasted_models=[]
	l_list = len(selected_models)
	i = 1
	with concurrent.futures.ProcessPoolExecutor(max_workers=int(n_cpu/2)) as executor:
		for obj_model in selected_models:
			print(f"Batch forecasting: start .. {i}/{l_list}", end='\r', flush=True)
			model = obj_model[2]['result']
			i += 1
			forecasted_models.append(executor.submit(forecast_n_step_forward, model, df_to_fcst, column_name, ci, f_w, refit))
		i = 1
		for f_model in concurrent.futures.as_completed(forecasted_models):
			print(f"Batch forecasting: done  ... {i}/{l_list}", end='\r', flush=True)
			i += 1
	
	#
	## Construction du dictionaire contenant tous les modeles entraînés
	#
	for model, f_model in zip(selected_models, forecasted_models):
		model[2]['forecast'] = f_model.result()
	t_end = perf_counter()
	print(f"\nLe batch forecasting a duré {t_end - t_start} secondes.")    

 

def forecast_evaluation(serie, model, criterion=['SE', 'SSE', 'MSE', 'RMSE'], SEplot=False):
	"""
	__Description__:
	  ...
	__Parametres__:
	  ...
	__Return__:
	  ...
	"""
	if not (isinstance(serie, pd.Series) or isinstance(serie, pd.DataFrame)):
		print("serie doit être de type Pandas.Series ou Pandas.DataFrame.")
		return
	if not isinstance(model, dict):
		print("model doit etre un dictionaire contenant les clefs 'model', 'fit', 'predict', 'forecast' ... .")
		return
	if not isinstance(criterion, list):
		print("criterion doit etre une liste contenant les differents criteres d'evaluation.")
		return
	if serie.shape[0] != model['forecast'].shape[0]:
		print("serie et model['forecast'] n'ont pas le meme nombre de lignes.")
		return
	f_eval={}
	if 'SE' in criterion:
		f_eval['SE'] = square(serie.values - model['forecast'].iloc[:,0])
	if 'SSE' in criterion:
		f_eval['SSE'] = sum(f_eval['SE'])
	if 'MSE' in criterion:
		f_eval['MSE'] = mean(f_eval['SSE'])
	if 'RMSE' in criterion:
		f_eval['RMSE'] = sqrt(f_eval['MSE'])
	if SEplot:
		#f_eval['SE plot'] = plt_SE(f_eval['SE'])
		pass
	return(f_eval)

 

def test_evaluation(serie, model, criterion=['SE', 'SSE', 'MSE', 'RMSE'], SEplot=False):
	"""
	__Description__:
	  ...
	__Parametres__:
	  ...
	__Return__:
	  ...
	"""
	if not (isinstance(serie, pd.Series) or isinstance(serie, pd.DataFrame)):
		print("serie doit être de type Pandas.Series ou Pandas.DataFrame.")
		return
	if not isinstance(model, dict):
		print("model doit etre un dictionaire contenant les clefs 'model', 'fit', 'predict', 'forecast' ... .")
		return
	if not isinstance(criterion, list):
		print("criterion doit etre une liste contenant les differents criteres d'evaluation.")
		return
	if serie.shape[0] != model['predict_test'].shape[0]:
		print("serie et model['predict_test'] n'ont pas le meme nombre de lignes.")
		return
	f_eval={}
	if 'SE' in criterion:
		f_eval['SE'] = square(serie.values - model['predict_test'].iloc[:,0])
	if 'SSE' in criterion:
		f_eval['SSE'] = sum(f_eval['SE'])
	if 'MSE' in criterion:
		f_eval['MSE'] = mean(f_eval['SSE'])
	if 'RMSE' in criterion:
		f_eval['RMSE'] = sqrt(f_eval['MSE'])
	if SEplot:
		#f_eval['SE plot'] = plt_SE(f_eval['SE'])
		pass
	return(f_eval)

 

def select_best_model(benchmark, on='predict', n_top=10, criteria='AIC'):
	"""
	__Description__:
	  La fonction permet de selectionner les n_top premier modèles
		regroupés dans l'instance benchmark en se basant sur le critère
		'criteria'.
		
	__Parametres__:
	  ...
	  
	___Return__:
	  sorted_models[:n_top]: [list] portion de liste correspondant au n_top premier
		  modeles en se basant sur le critere criteria
	"""
	if not isinstance(benchmark, dict):
		print("'benchmark' n'est pas un objet de type dictionnaire.")
		return
	first_key = list(benchmark.keys())[0]
	if not (on in benchmark[first_key].keys()):
		print(f"Il n'y a pas de clef '{on}' dans le dictionnaire 'benchmark'.")
		return
	if not (criteria in ['SSE', 'AIC', 'AICc', 'BIC', 'MSE', 'RMSE', 'R2']):
		print(f"{criteria} n'est pas une valeur accpetée pour le paramètre 'criteria'.")
		return
	models = sorted(benchmark, key=lambda x: benchmark[x][on][criteria])
	#sort_models = []
	sort_models = {}
	rank = 1
	for model in models[:n_top]:
		#sort_models.append([rank, model, benchmark[model]])
		sort_models[f'rank {rank}'] = benchmark[model]
		rank += 1
	
	return sort_models