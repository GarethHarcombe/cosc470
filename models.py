

class Model:

    def __init__(self, inference_func, params, requires_training=True, epochs=10):
        self.inference_func = inference_func
        self.params = params
        self.requires_training = requires_training
        self.epochs = epochs
        
    def forward(self, x):
        return self.inference_func(x, self.params)
        


def find_200(params):
        pass



class SimpleModel(Model):

    def __init__(self, inference_func, params, requires_training=True, epochs=10):
        super().__init__(inference_func, params, requires_training, epochs)
        self.inference_func = find_200


