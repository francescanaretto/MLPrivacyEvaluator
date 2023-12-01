from bbox import AbstractBBox

'''
In this file are contained the wrappers which implement the AbstractBbox class and methods.
If you don't find the wrapper for your model here, you can add yours by extending the abstract class AbstractBbox.
'''

'''The wrapper for the Scikit-learn models, such as Random Forest, Decision Trees etc.'''
class sklearn_classifier_wrapper(AbstractBBox):
    def __init__(self, classifier):
        super().__init__(classifier)


    def model(self):
        return self.bbox

    def predict(self, X):
        return self.bbox.predict(X)

    def predict_proba(self, X):
        return self.bbox.predict_proba(X)


#TODO Add the wrappers for the pytorch models and the keras models.


