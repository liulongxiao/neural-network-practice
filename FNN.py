import numpy as np
import numpy
import theano
import theano.tensor as T

class sigmoid:
    def __init__(self):
        self.x=T.dscalar()
        self.y=1/(1+T.exp(-self.x))
        self.func=theano.function([self.x],self.y)
        self.grad=theano.grad(self.y,self.x)
        self.grad_func=theano.function([self.x],self.grad)

class least_square:
    def __init__(self):
        self.x=T.dvector()
        self.y=T.dvector()
        self.z=T.dot(self.x-self.y,self.x-self.y)
        self.func=theano.function([self.x,self.y],self.z)
        self.grad = theano.grad(self.z, self.x)
        self.grad_func = theano.function([self.x,self.y], self.grad)

class neuron:
    def __init__(self,active_function):
        self.active_function=active_function
        self.input=None
        self.output=None
        self.params=None

    def cal_init(self,d):
        self.params=numpy.random.randn(d)/np.sqrt(d)

    def inform(self,input):
        self.input=input
        self.input_size=len(input)

    def forward(self):
        if  self.input is None:
            raise TypeError('layers forward before being informed')
        self.output=self.active_function.func(self.input.dot(self.params))

    def back_inform(self,back_input):
        self.back_input=back_input

    def backward(self):
        self.out_gradient=np.sum(self.back_input)
        self.before_active_gradient=self.out_gradient*self.active_function.grad_func(self.input.dot(self.params))
        self.params_gradient=np.zeros(self.input_size)
        for i in range(self.input_size):
            self.params_gradient[i]=self.before_active_gradient*self.input[i]
        self.back_output=self.before_active_gradient*self.params

    def update(self,learning_rate):
        self.params=self.params-self.params_gradient*learning_rate

class layer:
    def __init__(self,input_size,output_size):
        self.neurons=[]
        self.input_size=input_size
        self.output_size=output_size
        self.output=None

    def add_neurons(self,neurons):
        if len(neurons)!=self.output_size:
            raise IndexError('output_size does not match numbers of neurons ')
        for _neuron in neurons:
            _neuron.cal_init(self.input_size)
            self.neurons.append(_neuron)

    def forward(self,input):
        input=np.array(input)
        self.output = []
        for _neuron in self.neurons:
            _neuron.inform(input)
            _neuron.forward()
            self.output.append(_neuron.output)
        self.output=np.array(self.output)
    def backward(self,backward_input):
        backward_input=np.array(backward_input)
        self.backward_input=backward_input
        self.back_output=[]
        for i,_neuron in enumerate(self.neurons):
            self.neurons[i].back_inform(backward_input[i,:])
            _neuron.backward()
            self.back_output.append(_neuron.back_output)
        self.back_output=np.array(self.back_output).T

    def update(self,learning_rate):
        for neuron_ in self.neurons:
            neuron_.update(learning_rate)

class output_layer(layer):
    def __init__(self,input_size,output_size,loss_func):
        super().__init__(input_size,output_size)
        self.loss_func=loss_func
    def loss_(self,y):
        if self.output is None:
            raise ValueError('loss must be calculated after forward')
        self.loss=self.loss_func.func(self.output,y)

    def backward(self,y):
        self.loss_(y)
        self.back_input=self.loss_func.grad_func(self.output,y)
        self.back_output=[]
        for i, _neuron in enumerate(self.neurons):
            self.neurons[i].back_inform(self.back_input[i])
            _neuron.backward()
            self.back_output.append(_neuron.back_output)
        self.back_output = np.array(self.back_output).T

class graph:
    def __init__(self, hiden_layer_size):
        self.hiden_layer_size = hiden_layer_size
        self.layers=[]
    def add_layers(self,input_and_hiden_layers):
        if len(input_and_hiden_layers)!=self.hiden_layer_size+1:
            raise ValueError('input_and_hiden_layer_size dont match the number of hidenlayers')
        self.layers=input_and_hiden_layers
    def add_output_layer(self,output_layer_):
        self.output_layer_=output_layer_
    def check_layers_size(self):
        for i,thelayer in enumerate(self.layers[:-1]):
            if len(thelayer.neurons)!=self.layers[i].input_size:
                raise ValueError('layers size in graph not consistent')
        if len(self.layers[-1].neurons)!=self.output_layer.input_size:
            raise ValueError('output_layers size in graph not consistent')
    def forward(self,input):
        self.raw_input=np.array(input)
        for i in range(len(self.layers)):
            self.layers[i].forward(input)
            input=self.layers[i].output
        self.output_layer_.forward(input)
        self.output=self.output_layer_.output

    def backward(self,y):

        self.target=y
        self.output_layer_.backward(y)
        back_input =self.output_layer_.back_output
        for i in range(self.hiden_layer_size+1):
            self.layers[self.hiden_layer_size-i].backward(back_input)
            back_input=self.layers[self.hiden_layer_size-i].back_output

    def SCGD_update(self,learning_rate,x,y):
        beforeloss=self.output_layer_.loss
        for layer in self.layers:
            layer.update(learning_rate)
        self.output_layer_.update(learning_rate)
        self.forward(x)
        self.backward(y)
        afterloss=self.output_layer_.loss
    
    def update(self,learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)
        self.output_layer_.update(learning_rate)
        self.forward(self.raw_input)
        self.backward(self.target)
        
    
    def predict(self,X):
        X=np.array(X)
        if len(X.shape)==1:
            self.forward(X)
            predicted = self.output
            return predicted

        N=X.shape[0]
        predicted=np.zeros([N,self.output_layer_.output_size])
        for i in range(N):
            self.forward(X[i])
            predicted[i]=self.output
        return predicted

    def SCGD_train(self,X,Y,learning_rate=0.1):
        N=X.shape[0]
        for i in range(N):
            self.SCGD_update(learning_rate,X[i],Y[i])

ne1=neuron(sigmoid())
ne2=neuron(sigmoid())
ne3=neuron(sigmoid())
ne4=neuron(sigmoid())
ne5=neuron(sigmoid())
ne6=neuron(sigmoid())
la1=layer(3,2)
la1.add_neurons([ne1,ne2])
la2=layer(2,3)
la2.add_neurons([ne3,ne4,ne5])
la3=output_layer(3,1,least_square())
la3.add_neurons([ne6])
gr=graph(1)
gr.add_layers([la1,la2])
gr.add_output_layer(la3)
gr.forward([1,2,3])
gr.backward([1])
gr.predict([1,2,3])
gr.update(1)
for i in range(5000):
    gr.update(1)
    print(gr.output_layer_.loss)
