# Autogenerated by nbdev

d = { 'settings': { 'branch': 'master',
                'doc_baseurl': 'https://puhazoli.github.io/asbe/',
                'doc_host': 'https://puhazoli.github.io',
                'git_url': 'https://github.com/puhazoli/asbe',
                'lib_path': 'asbe'},
  'syms': { 'asbe.base': { 'asbe.base.BaseAcquisitionFunction': ('base.html#baseacquisitionfunction', 'asbe/base.py'),
                           'asbe.base.BaseAcquisitionFunction.__init__': ('base.html#baseacquisitionfunction.__init__', 'asbe/base.py'),
                           'asbe.base.BaseAcquisitionFunction.calculate_metrics': ( 'base.html#baseacquisitionfunction.calculate_metrics',
                                                                                    'asbe/base.py'),
                           'asbe.base.BaseAcquisitionFunction.select_data': ( 'base.html#baseacquisitionfunction.select_data',
                                                                              'asbe/base.py'),
                           'asbe.base.BaseActiveLearner': ('base.html#baseactivelearner', 'asbe/base.py'),
                           'asbe.base.BaseActiveLearner.__init__': ('base.html#baseactivelearner.__init__', 'asbe/base.py'),
                           'asbe.base.BaseActiveLearner._select_counterfactuals': ( 'base.html#baseactivelearner._select_counterfactuals',
                                                                                    'asbe/base.py'),
                           'asbe.base.BaseActiveLearner._update_dataset': ('base.html#baseactivelearner._update_dataset', 'asbe/base.py'),
                           'asbe.base.BaseActiveLearner.fit': ('base.html#baseactivelearner.fit', 'asbe/base.py'),
                           'asbe.base.BaseActiveLearner.plot': ('base.html#baseactivelearner.plot', 'asbe/base.py'),
                           'asbe.base.BaseActiveLearner.predict': ('base.html#baseactivelearner.predict', 'asbe/base.py'),
                           'asbe.base.BaseActiveLearner.query': ('base.html#baseactivelearner.query', 'asbe/base.py'),
                           'asbe.base.BaseActiveLearner.score': ('base.html#baseactivelearner.score', 'asbe/base.py'),
                           'asbe.base.BaseActiveLearner.simulate': ('base.html#baseactivelearner.simulate', 'asbe/base.py'),
                           'asbe.base.BaseActiveLearner.teach': ('base.html#baseactivelearner.teach', 'asbe/base.py'),
                           'asbe.base.BaseAssignmentFunction': ('base.html#baseassignmentfunction', 'asbe/base.py'),
                           'asbe.base.BaseAssignmentFunction.__init__': ('base.html#baseassignmentfunction.__init__', 'asbe/base.py'),
                           'asbe.base.BaseAssignmentFunction.select_treatment': ( 'base.html#baseassignmentfunction.select_treatment',
                                                                                  'asbe/base.py'),
                           'asbe.base.BaseDataGenerator': ('base.html#basedatagenerator', 'asbe/base.py'),
                           'asbe.base.BaseDataGenerator.__getitem__': ('base.html#basedatagenerator.__getitem__', 'asbe/base.py'),
                           'asbe.base.BaseDataGenerator.get_X': ('base.html#basedatagenerator.get_x', 'asbe/base.py'),
                           'asbe.base.BaseDataGenerator.get_data': ('base.html#basedatagenerator.get_data', 'asbe/base.py'),
                           'asbe.base.BaseDataGenerator.get_t': ('base.html#basedatagenerator.get_t', 'asbe/base.py'),
                           'asbe.base.BaseDataGenerator.get_y': ('base.html#basedatagenerator.get_y', 'asbe/base.py'),
                           'asbe.base.BaseITEEstimator': ('base.html#baseiteestimator', 'asbe/base.py'),
                           'asbe.base.BaseITEEstimator.__init__': ('base.html#baseiteestimator.__init__', 'asbe/base.py'),
                           'asbe.base.BaseITEEstimator._fit_ps_model': ('base.html#baseiteestimator._fit_ps_model', 'asbe/base.py'),
                           'asbe.base.BaseITEEstimator._predict_bin_or_con': ( 'base.html#baseiteestimator._predict_bin_or_con',
                                                                               'asbe/base.py'),
                           'asbe.base.BaseITEEstimator.fit': ('base.html#baseiteestimator.fit', 'asbe/base.py'),
                           'asbe.base.BaseITEEstimator.predict': ('base.html#baseiteestimator.predict', 'asbe/base.py'),
                           'asbe.base.BaseITEEstimator.prepare_data': ('base.html#baseiteestimator.prepare_data', 'asbe/base.py'),
                           'asbe.base.BaseStoppingRule': ('base.html#basestoppingrule', 'asbe/base.py'),
                           'asbe.base.BaseStoppingRule.__init__': ('base.html#basestoppingrule.__init__', 'asbe/base.py'),
                           'asbe.base.BaseStoppingRule.check_rule': ('base.html#basestoppingrule.check_rule', 'asbe/base.py'),
                           'asbe.base.FitTask': ('base.html#fittask', 'asbe/base.py'),
                           'asbe.base.FitTask.__init__': ('base.html#fittask.__init__', 'asbe/base.py')},
            'asbe.core': {},
            'asbe.estimators': { 'asbe.estimators.CausalForestEstimator': ('estimators.html#causalforestestimator', 'asbe/estimators.py'),
                                 'asbe.estimators.CausalForestEstimator.fit': ( 'estimators.html#causalforestestimator.fit',
                                                                                'asbe/estimators.py'),
                                 'asbe.estimators.CausalForestEstimator.predict': ( 'estimators.html#causalforestestimator.predict',
                                                                                    'asbe/estimators.py'),
                                 'asbe.estimators.GPEstimator': ('estimators.html#gpestimator', 'asbe/estimators.py'),
                                 'asbe.estimators.GPEstimator.predict': ('estimators.html#gpestimator.predict', 'asbe/estimators.py'),
                                 'asbe.estimators.OPENBTITEEstimator': ('estimators.html#openbtiteestimator', 'asbe/estimators.py'),
                                 'asbe.estimators.OPENBTITEEstimator.predict': ( 'estimators.html#openbtiteestimator.predict',
                                                                                 'asbe/estimators.py')},
            'asbe.helper': {'asbe.helper.get_ihdp_dict': ('misc.html#get_ihdp_dict', 'asbe/helper.py')},
            'asbe.models': { 'asbe.models.EMCMAcquisitionFunction': ('models.html#emcmacquisitionfunction', 'asbe/models.py'),
                             'asbe.models.EMCMAcquisitionFunction.__init__': ( 'models.html#emcmacquisitionfunction.__init__',
                                                                               'asbe/models.py'),
                             'asbe.models.EMCMAcquisitionFunction.calculate_metrics': ( 'models.html#emcmacquisitionfunction.calculate_metrics',
                                                                                        'asbe/models.py'),
                             'asbe.models.MajorityAssignmentFunction': ('models.html#majorityassignmentfunction', 'asbe/models.py'),
                             'asbe.models.MajorityAssignmentFunction.select_treatment': ( 'models.html#majorityassignmentfunction.select_treatment',
                                                                                          'asbe/models.py'),
                             'asbe.models.PMajorityAssignmentFunction': ('models.html#pmajorityassignmentfunction', 'asbe/models.py'),
                             'asbe.models.PMajorityAssignmentFunction.__init__': ( 'models.html#pmajorityassignmentfunction.__init__',
                                                                                   'asbe/models.py'),
                             'asbe.models.PMajorityAssignmentFunction.select_treatment': ( 'models.html#pmajorityassignmentfunction.select_treatment',
                                                                                           'asbe/models.py'),
                             'asbe.models.RandomAcquisitionFunction': ('models.html#randomacquisitionfunction', 'asbe/models.py'),
                             'asbe.models.RandomAcquisitionFunction.calculate_metrics': ( 'models.html#randomacquisitionfunction.calculate_metrics',
                                                                                          'asbe/models.py'),
                             'asbe.models.RandomAssignmentFunction': ('models.html#randomassignmentfunction', 'asbe/models.py'),
                             'asbe.models.RandomAssignmentFunction.__init__': ( 'models.html#randomassignmentfunction.__init__',
                                                                                'asbe/models.py'),
                             'asbe.models.RandomAssignmentFunction.select_treatment': ( 'models.html#randomassignmentfunction.select_treatment',
                                                                                        'asbe/models.py'),
                             'asbe.models.TypeSAcquistionFunction': ('models.html#typesacquistionfunction', 'asbe/models.py'),
                             'asbe.models.TypeSAcquistionFunction.calculate_metrics': ( 'models.html#typesacquistionfunction.calculate_metrics',
                                                                                        'asbe/models.py'),
                             'asbe.models.UncertaintyAcquisitionFunction': ('models.html#uncertaintyacquisitionfunction', 'asbe/models.py'),
                             'asbe.models.UncertaintyAcquisitionFunction.calculate_metrics': ( 'models.html#uncertaintyacquisitionfunction.calculate_metrics',
                                                                                               'asbe/models.py'),
                             'asbe.models.UncertaintyAssignmentFunction': ('models.html#uncertaintyassignmentfunction', 'asbe/models.py'),
                             'asbe.models.UncertaintyAssignmentFunction.select_treatment': ( 'models.html#uncertaintyassignmentfunction.select_treatment',
                                                                                             'asbe/models.py')}}}
