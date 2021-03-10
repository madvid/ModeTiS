# -------------------------------------------------------------------------- #
# Ce fichier constitue un module contenant les focntions permettant
# d'explorer l'espace des modeles SARIMAX
# -------------------------------------------------------------------------- #

##
### Importation des differents modules, fonctions
##

# Pour le multiprocessing
import concurrent.futures
from multiprocessing import cpu_count

# Pour la creation de l'ensemble des arrangements de coefficie itertools import product

# Pour mesurer la performance temporelle
from time import perf_counter

#
from numpy import log, arange, mean, square, sqrt, sum
from pandas import DataFrame, Series, concat
from pandas.core.dtypes.dtypes import PeriodDtype

# Fonctions relatives à l'analyse en serie temporelle
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.model_selection import TimeSeriesSplit

# Importation de modules maison
from package_tools.stats import *
from package_tools.plot import *


def purged_Kfold_sets(train, k, purge, embargo = 0):
	"""
	__Description__:
	  Implémentation de la méthode de Kfold avec purge et embargo
	  Le Kfold avec purge consiste en la présence d'un intervalle de valeurs
	  interdite entre l'ensemble de d'entrainement et de validation (un certain
	  nombre de points entre entrainement et validation sont retirés)
	
	__Parametres__:
	  train   : dataframe ou serie représentant l'ensemble d'entrainement
	  k       : le nombre de fold souhaité
	  purge   : la largeur de la période de purge (assez grande pour eviter la
				fuite d'information entre train et validation)
	  embargo : période similaire à la purge mais pour que la train et la validation
				soit suffisament éloigné dans le temps pour ne pas être sur des tendances
				similaires de la série, à l'interface des 2.
	
	__Return__:
	  folds   : (dict) dictonnaire de l'ensemble des folds. Chaque fold est une liste de
				2 dataframes, le dataframe d'entrainement et le dataframe de validation.
				folds['fold_1'] == [df_train, df_validation]
	"""
	if not (isinstance(train, pd.DataFrame) or isinstance(train, pd.Series)):
		raise Exception("Argument train n'est pas un pandas.DataFrame ou pandas.Series.")
	if not (isinstance(k, int) or isinstance(purge, int) or isinstance(embargo, int)):
		raise Exception("Argument k/purge/embargo n'est pas un int.")
	if not (k >= 1 and k <= 10):
		#
		## (to do: tester et améliorer et ne plus clipper k)
		## mais le clippage est probablement nécessaire pour les int négatifs et
		## des valeurs trop importantes
		#
		raise Exception("La valeur de k doit être entre 1 et 10 inclut.")
	if not (purge >= 0 and purge <= 20):
		#
		## (to do: tester et améliorer et ne plus clipper purge)
		## mais le clippage est probablement nécessaire pour les int négatifs et
		## des valeurs trop importantes
		#
		raise Exception("La valeur de purge doit être entre 1 et 10 inclut.")
	if not (embargo >= 0 and embargo <= 100):
		#
		## (to do: tester et améliorer et ne plus clipper embargo)
		## mais le clippage est probablement nécessaire pour les int négatifs et
		## des valeurs trop importantes
		#
		raise Exception("La valeur de embargo doit être entre 1 et 10 inclut.")
	
	#
	## Enregistrement de l'index d'origine et indexation sur un RangeInt
	#
	idx_name = train.index.name
	df = train.reset_index(inplace=False)
	
	folds={}
	l = train.shape[0]
	if l < 100:
		raise Exception("Le nombre de points au sein des données "
					   + "est trop faible pour envisager un Kfold.")
	
	#
	## definition des tailles du validation set en fonction des cas (debut/fin ou milieu)
	#
	l_valid_fold_1, reste_1 = int((l - 2*purge - embargo)/k), (l - 2*purge) % k
	l_valid_fold_2, reste_2 = int((l - 3*purge - embargo)/k), (l - 3*purge) % k
	l_valid_fold_1 += reste_1
	l_valid_fold_2 += reste_2
	sets = {}
	i = 1
	#
	## Cas du fold du début
	#
	folds[f'Fold_{i}'] = [df.loc[l_valid_fold_1 + purge + embargo:l-purge-1],
						 df.loc[0:l_valid_fold_1 - 1]]
	i += 1
	#
	## Cas des folds du 'milieu'
	#
	for i in range(2,k):
		end_E1 =  (i - 1) * l_valid_fold_2 - 1
		start_V = (i - 1) * l_valid_fold_2 + purge
		end_V = i * l_valid_fold_2 + purge - 1
		start_E2 = i * l_valid_fold_2 + 2 * purge + embargo
		end_E2 = l-purge
		E = pd.concat((df.loc[0:end_E1],df.loc[start_E2:end_E2]), axis = 0)
		V = df.loc[start_V:end_V]
		folds[f'Fold_{i}'] = [E, V]
		i += 1
	#
	## Cas du fold de fin:
	#
	folds[f'Fold_{i}'] = [df.loc[0:(i - 1) * l_valid_fold_1 - 1],
						 df.loc[(i - 1) * l_valid_fold_1 + purge : l-purge]]
	
	#
	## Reindexation sur l'index d'origine
	#
	if idx_name != None:
		for key in folds.keys():
			for df_i in folds[key]:
				df_i.set_index(idx_name, inplace=True)
	return(folds)


def model_fit(dct_model, serie, orders, s_orders, **kwargs):
	"""
	__Description__:
	
	__Parametres__:
		serie   : La Serie ou DataFrame constituant la serie temporelle
				  pour l'ajustement uniquement.
		orders  : un tuple de 3 int (p, d, q)
		s_orders: un tuple de 4 int (P, D, Q, S)
				  le complement est utilisé pour la partie test
		kwargs  : parametres supplementaires pour le modele qui sont:
		 -b_fit   : booléen précisant si l'ajustement du modele doit etre fait
		
	__Return__:
	Un dictionnaire constitué des éléments suivants:
		model: retour de la methode statsmodels.tsa.statespace.SARIMAX
		result_model: retour de la methode statsmodels.tsa.statespace.SARIMAX.fit
		statistique: valeurs des grandeurs stat AIC, SSE
	"""
	#
	## Protection basique contre une mauvaise definition des parametres
	#
	if not (isinstance(serie, DataFrame) or isinstance(serie, Series)) or not serie.dtype == 'float':
		print("Le parametre 'serie' n'est pas de type pandas.core.series.Series "
			   +"ou bien les objets de la serie ne sont pas de type 'float'.")
		return
	if not isinstance(serie.index.dtype, PeriodDtype):
		print("Il est preferable que l'index de la serie soit de type period[B], "
			   +"sinon il risque d'y avoir des predictions pour les samedis et dimanches.")
		return
	
	#
	## Reccuperation des differents parametres
	#
	pdq_params = orders
	PDQS_params = s_orders
	
	b_fit =kwargs.get('b_fit', True)
	stationarity=kwargs.get('stationarity', True)
	invertibility=kwargs.get('invertibility', True)

 
	#
	## Protection contre la mauvaise definition de pdq et PDQS ainsi que 
	## la mauvaise definition de stationarity et invertibility
	#
	if pdq_params == (0,0,0) and PDQS_params == (0,0,0,0):
		print("Aucune valeur pour 'order' et 'seasonal_order' n'ont été donnée.")
		return
	if not (isinstance(pdq_params, list) or isinstance(pdq_params, tuple)):
		print("'pdq_params' correspondant aux ordres non-saisonnier du modèle SARIMA ne sont pas "
			 +"au bon format.")
		return
	if not isinstance(b_fit, bool):
		print("'b_fit' parameter must be a boolean.")
		return
	if (not isinstance(stationarity, bool) or not isinstance(invertibility, bool)):
		print("'stationarity' et/ou 'invertibility' ne sont pas des booléens:")
		return
	
	#
	## declaration du modele et stockage du modele dans le dictionnaire de résultat 
	#
	if (pdq_params[0] == 0) and (pdq_params[2] == 0):
		raise Exception("(S)ARIMA model parameters issue: p = 0 and q = 0")
	model = SARIMAX(serie,
					order=pdq_params, seasonal_order=PDQS_params,
					enforce_stationarity=stationarity,
					enforce_invertibility=invertibility)
	dct_model['model'] = model
	if b_fit:
		result_model = model.fit(maxiter=400)
		dct_model['result'] = result_model
	return

 

def model_eval(dct_model, **kwargs):
	"""
	__Description__:
		...
	__Parametres__:
		kwargs  : parametres supplementaires pour le modele qui sont:
		 -b_eval  : booléen précisant si l'évaluation du modele sur train doit etre fait
		
	__Return__:
	Un dictionnaire constitué des éléments suivants:
		model: retour de la methode statsmodels.tsa.statespace.SARIMAX
		result_model: retour de la methode statsmodels.tsa.statespace.SARIMAX.fit
		statistique: valeurs des grandeurs stat AIC, SSE
	"""
	b_eval=kwargs.get('b_eval', True)
	if not isinstance(b_eval, bool):
		print("'b_eval' parameter must be a boolean.")
		return 'evaluation'
	if 'result' not in dct_model.keys():
		print("No SARIMAXResult present for the key 'result' in dct_model.")
		return 'evaluation'
	if b_eval:
		eval_stat = {}
		eval_stat['AIC'] = dct_model['result'].aic
		eval_stat['BIC'] = dct_model['result'].bic
		eval_stat['SSE'] = dct_model['result'].sse
		eval_stat['MSE'] = dct_model['result'].mse
		eval_stat['LjungBox test'] = acorr_ljungbox(x=dct_model['result'].resid,
										 lags=[int(log(dct_model['result'].resid.shape[0]))])
		dct_model['eval_stat']=dict(eval_stat)
	return

 

def model_pred(dct_model, serie, **kwargs):
	"""
	__Description__:
	
	__Parametres__:
		serie   : La Serie ou DataFrame constituant la serie temporelle
				  pour l'ajustement uniquement.
		kwargs  : parametres supplementaires pour le modele qui sont:
		 -b_pred  : booléen précisant si la prediction doit etre fait
		
	__Return__:
	Un dictionnaire constitué des éléments suivants:
		model: retour de la methode statsmodels.tsa.statespace.SARIMAX
		result_model: retour de la methode statsmodels.tsa.statespace.SARIMAX.fit
		statistique: valeurs des grandeurs stat AIC, SSE
	"""
	if not (isinstance(serie, DataFrame) or isinstance(serie, Series)) or not serie.dtype == 'float':
		print("Le parametre 'serie' n'est pas de type pandas.core.series.Series "
			   +"ou bien les objets de la serie ne sont pas de type 'float'.")
		return
	
	b_pred=kwargs.get('b_pred', True)
	ci = kwargs.get('ci', 0.9)
	
	if not isinstance(b_pred, bool):
		print("'b_pred' parameter must be a boolean.")
		return 'prediction'
	
	if not isinstance(ci, float) and (ci < 0.5 or ci > 0.99):
		print("'ci' must be a float within 0.5 and 0.99.")
		return
	
	if 'result' not in dct_model.keys():
		print("No SARIMAXResult present for the key 'result' in dct_model.")
		return 'prediction'
	if b_pred:
		predict = get_prediction_values(dct_model['result'])
		ci_predict = get_prediction_conf_int(dct_model['result'], ci)
		df_predict = concat([predict, ci_predict], axis=1)
		df_predict.index = serie.index
		dct_model['predict']= df_predict
	return

 

def model_fcst(dct_model, serie, **kwargs):
	"""
	__Description__:
	
	__Parametres__:
		serie   : La Serie ou DataFrame constituant la serie temporelle
				  pour l'ajustement uniquement.
		kwargs  : parametres supplementaires pour le modele qui sont:
		 -b_fcst: booléen précisant si le forecast doit etre fait
		
	__Return__:
	...
	"""
	if not (isinstance(serie, DataFrame) or isinstance(serie, Series)) or not serie.dtype == 'float':
		print("Le parametre 'serie' n'est pas de type pandas.core.series.Series "
			   +"ou bien les objets de la serie ne sont pas de type 'float'.")
		return
	
	b_fcst=kwargs.get('b_fcst', True)
	ci = kwargs.get('ci', 0.9)
	if not isinstance(b_fcst, bool):
		print("'b_fcst' parameter must be a boolean.")
		return 'forecast'
	if not isinstance(ci, float) and (ci < 0.5 or ci > 0.99):
		print("'ci' must be a float within 0.5 and 0.99.")
		return 'forecast'
	
	if 'result' not in dct_model.keys():
		print("No SARIMAXResult present for the key 'result' in dct_model.")
		return 'forecast'
	dct_model['fcst'] = forecast_n_step_forward(dct_model['result'], serie, None, ci, 1, False)
	return

 

def model_val(dct_model, serie, **kwargs):
	"""
	__Description__:
	
	__Parametres__:
		serie   : La Serie ou DataFrame constituant la serie temporelle
				  pour l'ajustement uniquement.
		kwargs  : parametres supplementaires pour le modele qui sont:
		 -b_val  : booléen précisant si la validation doit etre fait
		
	__Return__:
		...
	"""
	if not (isinstance(serie, DataFrame) or isinstance(serie, Series)) or not serie.dtype == 'float':
		print("Le parametre 'serie' n'est pas de type pandas.core.series.Series "
			   +"ou bien les objets de la serie ne sont pas de type 'float'.")
		return
	
	b_val=kwargs.get('b_val', True)
	if not isinstance(b_val, bool):
		print("'b_val' parameter must be a boolean.")
		return 'validation'
	
	if 'result' not in dct_model.keys() or 'fcst' not in dct_model.keys():
		print("No SARIMAXResult present for the key 'result' or forecast values for 'fcst' in dct_model.")
		return 'validation'
	
	if 'validation_score' not in dct_model.keys():
		metrics = {}
		metrics['SSE'] = 0
		metrics['MSE'] = 0
		metrics['RMSE'] = 0
		metrics['AIC'] = 0
		metrics['AICc'] = 0
		metrics['BIC'] = 0
		dct_model['validation_score'] = metrics
	## SSE
	SSE = sum(square(serie.values - dct_model['fcst'].iloc[:,0]))
	dct_model['validation_score']['SSE'] += SSE
	## MSE
	MSE = mean(square(serie.values - dct_model['fcst'].iloc[:,0]))
	dct_model['validation_score']['MSE'] += MSE
	## RMSE
	RMSE = sqrt(MSE)
	dct_model['validation_score']['RMSE'] += RMSE
	## AIC - 
	n = dct_model['fcst'].shape[0]
	k = dct_model['model'].k_ar_params + dct_model['model'].k_ma_params
	k += dct_model['model'].k_seasonal_ar_params + dct_model['model'].k_seasonal_ma_params
	AIC = log(square(SSE / n)) + (n + 2 * k) / n 
	dct_model['validation_score']['AIC'] += AIC
	## AICc
	if (n - k - 2) != 0:
		AICc = log(square(SSE / n)) + (n + k) / (n - k - 2) 
		dct_model['validation_score']['AICc'] += AICc
	## BIC
	# BIC = formule du BIC
	# dct_model['validation_score']['BIC'] += BIC
	# R2
	# BIC = formule du R2
	# dct_model['validation_score']['R2'] += R2
	return

 

def wrapper_fit_pred_val(serie, order, s_order, **kwargs):
	dct_model={}
	stop_at = 'fit'
	tscv = TimeSeriesSplit(n_splits=3, test_size=10, gap=10)
	try:
		for train_idx, val_idx  in tscv.split(serie):
			print("ajustement:", f"[{serie.index[train_idx[0]]} --> {serie.index[train_idx[-1]]}]",
				  "-- validation:", f"[{serie.index[val_idx[0]]} --> {serie.index[val_idx[-1]]}]")
			s_train = serie.iloc[train_idx]
			s_valid = serie.iloc[val_idx]
			model_fit(dct_model, s_train, order, s_order, **kwargs)
			stop_at = model_eval(dct_model, **kwargs)
			stop_at = model_pred(dct_model, s_train, **kwargs)
			stop_at = model_fcst(dct_model, s_valid, **kwargs)
			stop_at = model_val(dct_model, s_valid, **kwargs)
			for key in dct_model["validation_score"].keys():
				dct_model['validation_score'][key] /= tscv.n_splits
	except:
		print(f"Une exception est survenue pour: SARIMA {order}{s_order} -- [{stop_at}]")
		dct_model = None
	return dct_model

 

def __construct_list_order_difference__(serie):
	"""
	__Description__:
	  La fonction effectue 2 tests afin d'estimer la stationnarité de la TS:
		* test de Dickey-Fuller augmenté
		* test de Kwiatkowski–Phillips–Schmidt–Shin
	  La fonction se base par défaut sur les valeurs critiques à 5% mais cela
	  peut être changé en modifiant la variable seuil (str: '1%','5%','10%')
	  ou via alpha (mettre une valeur pour la valeur p; pqr défaut: 0.05).
	  
	  Pour test ADF, la valeur p est relative à la vraissemblance de l'hypo-
	  -these H0 associée:
	  H0 = il y a une racine unitaire ... Ce qui nous intéresse est l'hypot-
	  -hèse alternative = la série est stationnaire ou stationnaire+tendance.
	  
	  Pour test KPSS, la valeur p est relative à la vraissemblance de l'hypo-
	  -these H0 associée:
	  H0 = la série est stationnaire avec une tendance, l'hypothèse alterna-
	  -tive est qu'il y a une racine unitaire.
	
	__Return__:
	  lst_d : [list] [0]/[1]/[0,1] les differentes valeurs pour d
	
	__Remarks__:
	  Pour comprendre les résultats des tests:
	  https://www.statsmodels.org/stable/examples/notebooks/generated/
	  stationarity_detrending_adf_kpss.html?highlight=stationarity
	  
	  Dans l'idée la fonction suit le principe suivant:
	  %%%%%%%%%%%%%%%%%%%(issu de la page juste avant)%%%%%%%%%%%%%%%%%%%%%%%%
		Case 1: Both tests conclude that the series is not stationary
				-> The series is not stationary
		Case 2: Both tests conclude that the series is stationary
				-> The series is stationary
		Case 3: KPSS indicates stationarity and ADF indicates non-stationarity
				-> The series is trend stationary.
				Trend needs to be removed to make series strict stationary.
				The detrended series is checked for stationarity.
		Case 4: KPSS indicates non-stationarity and ADF indicates stationarity
				-> The series is difference stationary.
				Differencing is to be used to make series stationary.
				The differenced series is checked for stationarity.
	  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	  
	  La valeur p correspond à un score de probabilité sur lequel nous pouvons
	  décider de rejeter où ou non. Si p est inférieur à un critère alpha pré-
	  -défini (typiquement 0.05), nous rejetons H0
	  
	  La statistique du test (ADF/KPSS) est une grandeur basée sur une formule
	  Pour rejeter H0, la valeur de la statistique doit être plus grande que 
	  la valeur critique considérée (et cela se reflète dans la faible valeur
	  de p)
	"""
	reg_values_adf = ['nc','c','ct','ctt']
	reg_values_kpss = ['c','ct']
	
	adf_results = []
	kpss_results = []
	
	ADF_seuil = '1%'
	KPSS_seuil = '10%'
	alpha_ADF = 1e-4
	alpha_KPSS = 1e-4 # /!\ p-value uniquement dans intervalle [0.01,0.1]
	
	for reg_val in reg_values_adf:
		adf_result = adfuller(serie, regression=reg_val)
		print("ADF__:", adf_result)
		adf_results.append([adf_result[0], adf_result[1], adf_result[4]])
	for reg_val in reg_values_kpss:
		kpss_result = kpss(serie, regression=reg_val, nlags='auto')
		print("KPSS__:", kpss_result)
		kpss_results.append([kpss_result[0], kpss_result[1], kpss_result[3]])
	
	stat_adf = False
	stat_kpss = False
	p_val_adf = False
	p_val_kpss = False
	
	for adf_elem in adf_results:
		if adf_elem[0] < adf_elem[2][ADF_seuil]:
			stat_adf = True
		if adf_elem[1] < alpha_ADF:
			p_val_adf = True
	for kpss_elem in kpss_results:
		if kpss_elem[0] < kpss_elem[2][KPSS_seuil]:
			stat_kpss = True
		if kpss_elem[1] > alpha_KPSS:
			p_val_kpss = True
	
	if stat_adf and stat_kpss and p_val_adf and p_val_kpss:
		print("[ADF] + [KPSS]: TS est stationnaire.")
		return ([0])
	if not (stat_adf and p_val_adf) and not (stat_kpss and p_val_kpss):
		print("[ADF] + [KPSS]: TS n'est pas stationnaire.")
		return ([-1])
	if (stat_adf and p_val_adf) and (not (stat_kpss and p_val_kpss)):
		print("[ADF]: TS stationnaire\n[KPSS] non stationnaire\n  --> Stationnaire apres differenciation")
		return ([1])
	if (not (stat_adf and p_val_adf)) and (stat_kpss and p_val_kpss):
		print("[KPSS]: TS stationnaire\n[ADF]: TS non stationnaire\n  --> Staionnaire avec tendance.")
		return ([1])

 

def __exclusion_rule_1__(orders, period):
	"""
	__Description__:
	  La fonction permet de retirer les arrangements de paramètres où au moins
	  un paramètre (AR ou MA) se retrouve à la fois dans la partie normale et
	  saisonnière.
	  
	___Return__:
	  Nothing
	"""
	loop_again = True
	while loop_again == True:
		loop_again = False
		for coeffs in orders:
			pdq_params, PDQS_params=coeffs
			if PDQS_params[1] != 0 and ((pdq_params[0] >= period) or (pdq_params[2] >= period)):
				orders.remove(coeffs)
				loop_again = True

 

def __exclusion_rule_2__(orders):
	"""
	__Description__:
	  La fonction permet de retirer les arrangements de paramètres où D est
	  égale à 0. En effet si D = 0 alors il n'y a tout simplement pas de
	  termes saisonniers.
	  
	___Return__:
	  Nothing
	"""
	for order in orders:
		if order[1][1] == 0:
			order[1][0] = 0
			order[1][2] = 0
			order[1][3] = 0

 

def __exclusion_rule_3__(orders):
	"""
	__Description__:
	  La fonction permet de retirer les arrangements de paramètres où p et q
	  sont nulles. En effet si p=0 et q = 0 alors il n'y a tout simplement pas de termes
	  "classiques".
	  
	___Return__:
	  Nothing
	"""
	loop_again = True
	while loop_again == True:
		loop_again = False
		for order in orders:
			n_orders, s_orders = order
			if (n_orders[0] == 0) and (n_orders[2] == 0):
				orders.remove(order)
				loop_again = True
	

 

def generate_params_SARIMA(serie, max_pdq, max_PDQ, period):
	"""
	__Description__:
	  Fonction permettant la génération des paramètres pour les modèles
	  SARIMA.
	__Parameters__:
	  max_pdq: un int correspondant au maximum que peut prendre p ou q
	  max_PDQ: un int correspondant au maximum que peut prendre P ou Q
	  period : un int correspondant au rang des termes periodiques
	__Return__:
	  orders: une liste de tuple à 2 éléments, le 1er pour les paramètres
			  pdq et le 2nd pour les paramètres PDQS.
	"""
	order=list(product((arange(max_pdq)), repeat=2))
	s_order=list(product((arange(max_PDQ)),repeat=2))
	lst_d = __construct_list_order_difference__(serie)
	##if lst_d[0] == -1:
	##    return None
	lst_d = [0, 1]
	lst_D = [0, 1]
	order=[[elem[0], d, elem[1]] for d in lst_d for elem in order]
	s_order=[[elem[0], D, elem[1], period] for D in lst_D for elem in s_order]
	orders=[order, s_order]
	orders= [elem for elem in product(*orders)]
	
	#
	## Regles d'exclustion de certains jeux de coefficients
	## Regle 1: Un terme ne peux pas etre a la fois dans la partie normale
	##    et dans la partie saisonniere
	## Regle 2: si D=0 alors il ne peut pas y avoir de termes saisonniers
	##    les ordres saisonniers sont tous mis à 0
	#
	__exclusion_rule_1__(orders, period)
	__exclusion_rule_2__(orders)
	__exclusion_rule_3__(orders)
	
	#
	## Elimination des repetitions apres exclusion_rule_2 en effet, apres la
	## fonction on peut avoir plusieurs fois le même SARIMA(p,d,q)(0,0,0,0)
	#
	orders = [(tuple(elem[0]),tuple(elem[1])) for elem in orders]
	orders = list(set(orders))
	
	return orders

 

def multi_grid_search_model(serie, max_pdq, max_PDQ, period, verbose=False):
	"""
	__Description__:
	  Fonction permettant l'exploration de modele dans l'espace dont les dimensions
	  sont les coefficients du modele SARIMAX. Cette fonction est la version
	  multiprocessing de grid_search_model.
	__Parameters__:
	  serie  : La Serie ou DataFrame constituant la serie temporelle.
	  max_pdq: un int correspondant au maximum que peut prendre p ou q
	  max_PDQ: un int correspondant au maximum que peut prendre P ou Q
	  period : un int correspondant au rang des termes periodiques
	__Return__:
	  benchmark: un dictionnaire contenant tous les modeles explorés
	"""
	t_start = perf_counter()
	
	#
	## Generation des uplets de parametres [(p,d,q)(P,D,Q,S)]
	#
	orders = generate_params_SARIMA(serie, max_pdq, max_PDQ, period)
	
	t_generation = perf_counter()
	print(f"{t_generation - t_start} secondes pour generer l'ensemble des combinaisaons des parametres.")
	orders_size=len(orders)
	
	#
	## Multiprocessing sur la procedure de declaration et d'ajustement
	#
	i = 1
	n_cpu = cpu_count()
	cpu_use = int(n_cpu/2)
	results=[]
	with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_use) as executor:
		for order, s_order in orders:
			print(f"... running ... modele SARIMA: {order}{s_order}.", end = '\r', flush=True)
			results.append(executor.submit(wrapper_fit_pred_val,
										   serie, order, s_order,
										   b_fit=True,
										   b_pred=True,
										   b_eval=False,
										   b_fcst=True,
										   b_val=True,
										   stationarity=True,
										   invertibility=False))
		
		for task in concurrent.futures.as_completed(results):
			if task.result() != None and 'model' in task.result().keys():
				pdq = task.result()['model'].order
				PDQS = task.result()['model'].seasonal_order
				print(f"... done ... modele SARIMA: {pdq}{PDQS} -- {i}/{orders_size}", end='\r', flush=True)
			i += 1
	
	#
	## Construction du dictionaire contenant tous les modeles entraînés
	#
	benchmark={}
	for idx, result in zip(range(1, len(orders)+1), results):
		if result.result() != None:
			benchmark[f"model {idx}"] = result.result()
	t_end = perf_counter()
	print(f"\nL'exploration a duré {t_end - t_start} secondes.")
	return (benchmark)

 

def grid_search_model(serie, max_pdq, max_PDQ, period, verbose=False):
	"""
	__Description__:
	  Fonction permettant l'exploration de modele dans l'espace dont les dimensions
	  sont les coefficients du modele SARIMAX
	__Parameters__:
	  serie  : La Serie ou DataFrame constituant la serie temporelle.
	  max_pdq: un int correspondant au maximum que peut prendre p ou q
	  max_PDQ: un int correspondant au maximum que peut prendre P ou Q
	  period : un int correspondant au rang des termes periodiques
	  
	__Return__:
	  benchmark: un dictionnaire contenant tous les modeles explorés
	"""
	t_start = perf_counter()
	#
	## Generation des uplets de parametres [(p,d,q)(P,D,Q,S)]
	#
	order=list(product((arange(max_pdq)), repeat=2))
	s_order=list(product((arange(max_PDQ)),repeat=2))
	lst_d = __construct_list_order_difference__(serie)
	#if lst_d[0] == -1:
	#    return None
	## Pour le moment on skip le cas ou les ADF et KPSS disent que la serie
	## n'est stationnaire
	lst_d[0, 1]
	lst_D = [0, 1]
	order=[[elem[0], d, elem[1]] for d in lst_d for elem in order]
	s_order=[[elem[0], D, elem[1], period] for D in lst_D for elem in s_order]
	orders=[order, s_order]
	orders= [elem for elem in product(*orders)]
	benchmark={}
	
	#
	## Regles d'exclustion de certains jeux de coefficients
	## Regle 1: Un terme ne peux pas etre a la fois dans la partie normale
	##    et dans la partie saisonniere
	## Regle 2: si D=0 alors il ne peut pas y avoir de termes saisonniers
	##    les ordres saisonniers sont tous mis à 0
	#
	__exclusion_rule_1__(orders, period)
	__exclusion_rule_2__(orders)
	
	#
	## Elimination des repetitions apres exclusion_rule_2 en effet, apres la
	## fonction on peut avoir plusieurs fois le même SARIMA(p,d,q)(0,0,0,0)
	#
	orders = [(tuple(elem[0]),tuple(elem[1])) for elem in orders]
	orders = list(set(orders))
	
	t_generation = perf_counter()
	print(f"{t_generation - t_start} secondes pour generer l'ensemble des combinaisaons des parametres.")
	orders_size=len(orders)
	i = 1
	
	for idx, coeffs in enumerate(orders, start=1):
		pdq_params, PDQS_params=coeffs
		if verbose:
			print(f"ordres (p,d,q):{pdq_params} -- ordres saisonniers:{PDQS_params} -- idx={idx}", end="\r")
		try:
			benchmark[f"modele {idx}"]=wrapper_model_fit_test(serie,
												  pdq_params, PDQS_params,
												  {'b_fit':True,
												   'b_eval':True,
												   'b_pred':True,
												   'b_fcst':True,
												   'b_val':True,
												   'stationarity':True,
												   'invertibility':True})
		except:
			  print(f"Une exception a eu lieu pour: order={pdq_params} -- s_order={PDQS_params} (idx={idx}).")
	
	t_end = perf_counter()
	print(f"\nL'exploration a duré {t_end - t_start} secondes.")
	return benchmark