import numpy as np
from . import tensor


class Module(object):
    """Base class for all neural network modules.
    """
    def __init__(self) -> None:
        """If a module behaves different between training and testing,
        its init method should inherit from this one."""
        self.training = True

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Defines calling forward method at every call.
        Should not be overridden by subclasses.
        """
        return self.forward(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Defines the forward propagation of the module performed at every call.
        Should be overridden by all subclasses.
        """
        ...

    def backward(self, dy: np.ndarray) -> np.ndarray:
        """Defines the backward propagation of the module.
        """
        return dy

    def train(self):
        """Sets the mode of the module to training.
        Should not be overridden by subclasses.
        """
        if 'training' in vars(self):
            self.training = True
        for attr in vars(self).values():
            if isinstance(attr, Module):
                Module.train()

    def eval(self):
        """Sets the mode of the module to eval.
        Should not be overridden by subclasses.
        """
        if 'training' in vars(self):
            self.training = False
        for attr in vars(self).values():
            if isinstance(attr, Module):
                Module.eval()


class Linear(Module):

    def __init__(self, in_length: int, out_length: int):
        """Module which applies linear transformation to input.

        Args:
            in_length: L_in from expected input shape (N, L_in).
            out_length: L_out from output shape (N, L_out).
        """
        
        # w[0] for bias and w[1:] for weight
        self.in_length = in_length 
        self.out_length = out_length
        self.w = tensor.tensor((in_length + 1, out_length))

    def forward(self, x):
        """Forward propagation of linear module.

        Args:
            x: input of shape (N, L_in).
        Returns:
            out: output of shape (N, L_out).
        """

        # TODO Implement forward propogation
        # of linear module.
        self.x = x
        self.length = x.shape[0]
        #self.results = np.dot(x, self.w[1:]) + np.tile(self.w[0], self.length).reshape(-1,1)
        self.results = np.dot(x, self.w[1:]) + np.tile(self.w[0], self.length).reshape(-1, self.out_length) 

        return self.results
        # End of todo

    def backward(self, dy):
        """Backward propagation of linear module.

        Args:
            dy: output delta of shape (N, L_out).
        Returns:
            dx: input delta of shape (N, L_in).
        """

        # TODO Implement backward propogation
        # of linear module.
        derivative = self.w[1:].T
        dx = np.dot(dy, derivative)
        
        weight_grad = np.dot(self.x.T, dy)
        #bias_grad = (self.w[0] * np.sum(dy, axis= 0)).reshape(1,self.out_length) #self.length
        bias_grad = np.sum(dy, axis= 0).reshape(1, self.out_length)
        
        #weight_grad = np.dot(self.x.T, dy)#/self.length
        #bias_grad = (self.w[0] * np.sum(dy, axis= 0)).reshape(1, self.out_length)#/self.length
        
        self.w.grad = np.concatenate((bias_grad, weight_grad))
        #self.w.grad = np.concatenate((bias_grad, weight_grad))
        return np.float16(dx)
        # End of todo


#what's the following modules about ?? 

class BatchNorm1d(Module):

    def __init__(self, length: int, momentum: float=0.9):
        """Module which applies batch normalization to input.

        Args:
            length: L from expected input shape (N, L).
            momentum: default 0.9.
        """
        super(BatchNorm1d, self).__init__()

        # TODO Initialize the attributes
        # of 1d batchnorm module.

        ...

        # End of todo

    def forward(self, x):
        """Forward propagation of batch norm module.

        Args:
            x: input of shape (N, L).
        Returns:
            out: output of shape (N, L).
        """

        # TODO Implement forward propogation
        # of 1d batchnorm module.

        ...

        # End of todo

    def backward(self, dy):
        """Backward propagation of batch norm module.

        Args:
            dy: output delta of shape (N, L).
        Returns:
            dx: input delta of shape (N, L).
        """

        # TODO Implement backward propogation
        # of 1d batchnorm module.

        ...

        # End of todo


class Conv2d(Module):

    def __init__(self, in_channels: int, channels: int, kernel_size: int=3,
                 stride: int=1, padding: int=0, bias: bool=True):
        """Module which applies 2D convolution to input.

        Args:
            in_channels: C_in from expected input shape (B, C_in, H_in, W_in).
            channels: C_out from output shape (B, C_out, H_out, W_out).
            kernel_size: default 3.
            stride: default 1.
            padding: default 0.
        """

        # TODO Initialize the attributes
        # of 2d convolution module.
        self.inShape = in_channels
        self.outShpae = channels
        self.kernel_size = kernel_size
        self.kernel = tensor.tensor((channels, self.kernel_size, self.kernel_size))
        #self.bias = tensor.tensor((self.kernel_size, self.kernel_size)) if bias else np.zeros((kernel_size, kernel_size))
        self.bias = tensor.tensor((channels, self.kernel_size, self.kernel_size)) if bias else np.zeros_like(self.kernel)
        self.stride = stride
        self.padding = padding
        
        ...

        # End of todo

    def forward(self, x):
        """Forward propagation of convolution module.

        Args:
            x: input of shape (B, C_in, H_in, W_in).
        Returns:
            out: output of shape (B, C_out, H_out, W_out).
        """
        
        
        # TODO Implement forward propogation
        # of 2d convolution module.
        b, c, h_0, w_0 = x.shape
        
        # padding section
        x_padding = []
        step = self.padding
        if step != 0:
            for image in x:
                # image.shape: channel * h * w
                d3_p = np.pad(image, step, 'constant')[step:-step]
                x_padding.append(d3_p)
            x_pad = np.array(x_padding).reshape((b, c, h_0+2*step, w_0+2*step))
        else: x_pad = x
        
        b, c, h, w = x_pad.shape
        n = self.kernel_size
        ks, stride = self.kernel_size, self.stride
        H_out = int((h - ks) / stride + 1)                                   
        W_out = int((w - ks) / stride + 1)     
        out = np.zeros((n, c, H_out, W_out))
        for i in range(H_out):
            for j in range(W_out):
                temp = x_pad[:,:, i*stride:i*stride+n, j*stride:j*stride+n]
                #x_masked = x[:, :, i * stride: i * stride + HH, j * stride: j * stride + WW]
                #out[:, :, i, j] = np.max(x_masked, axis=(2, 3))
                #for channel in self.kernel:
                temp = np.multiply(self.kernel, temp)
                out[i][j] += temp.sum()
        return out + self.bias

        # End of todo

    def backward(self, dy):
        """Backward propagation of convolution module.

        Args:
            dy: output delta of shape (B, C_out, H_out, W_out).
        Returns:
            dx: input delta of shape (B, C_in, H_in, W_in).
        """

        # TODO Implement backward propogation
        # of 2d convolution module.

        ...

        # End of todo


class Conv2d_im2col(Conv2d):

    def forward(self, x):

        # TODO Implement forward propogation of
        # 2d convolution module using im2col method.
        image = x
        imageCol = []
        for i in range(0, image.shape[0] - self.ks + 1, self.stride):
            for j in range(0, image.shape[1] - self.ks + 1, self.stride):
                col = image[i:i + self.ks, j:j + self.ks, :].reshape([-1])
                # col = image[:, i:i + self.ks, j:j + self.ks, :].reshape([-1])  # Do not use .view([-1])
                imageCol.append(col)
        imageCol = np.array(imageCol)  # shape: [(h*w),(c*h*w)] kernel's height, width and channels
        return imageCol

        # End of todo


class AvgPool(Module):

    def __init__(self, kernel_size: int=2,
                 stride: int=2, padding: int=0):
        """Module which applies average pooling to input.

        Args:
            kernel_size: default 2.
            stride: default 2.
            padding: default 0.
        """

        # TODO Initialize the attributes
        # of average pooling module.
        self.ks = kernel_size
        self.stride = stride
        self.padding = padding

        # End of todo

    def forward(self, x):
        """Forward propagation of average pooling module.

        Args:
            x: input of shape (B, C, H_in, W_in).
        Returns:
            out: output of shape (B, C, H_out, W_out).
        """

        # TODO Implement forward propogation
        # of average pooling module.
        b, c, h_0, w_0 = x.shape
        
        # padding section
        x_padding = []
        step = self.padding
        if step != 0:
            for image in x:
                # image.shape: channel * h * w
                d3_p = np.pad(image, step, 'constant')[step:-step]
                x_padding.append(d3_p)
            x_pad = np.array(x_padding).reshape((b, c, h_0+2*step, w_0+2*step))
        else: x_pad = x
        
        b, c, h, w = x_pad.shape
        out = np.zeros([b, c, h//self.stride, w//self.stride]) 
        self.index = np.zeros_like(x)
        for b in range(b):
            for d in range(c):
                for i in range(h//self.stride):
                    for j in range(w//self.stride):
                        _x = i *self.stride
                        _y = j *self.stride
                        out[b, d, i, j] = np.mean((x[b, d, _x:_x+self.ks, _y:_y+self.ks]))
        return out
        # End of todo

    def backward(self, dy):
        """Backward propagation of average pooling module.

        Args:
            dy: output delta of shape (B, C, H_out, W_out).
        Returns:
            dx: input delta of shape (B, C, H_in, W_in).
        """

        # TODO Implement backward propogation
        # of average pooling module.

        return np.repeat(np.repeat(dy, self.stride, axis=2), self.stride, axis=3)/(self.ks * self.ks)

        # End of todo


class MaxPool(Module):

    def __init__(self, kernel_size: int=2,
                 stride: int=2, padding: int=0):
        """Module which applies max pooling to input.

        Args:
            kernel_size: default 2.
            stride: default 2.
            padding: default 0.
        """

        # TODO Initialize the attributes
        # of maximum pooling module.
        
        self.ks = kernel_size
        self.stride = stride
        self.padding = padding
        # End of todo

    def forward(self, x):
        """Forward propagation of max pooling module.

        Args:
            x: input of shape (B, C, H_in, W_in).
        Returns:
            out: output of shape (B, C, H_out, W_out).
        """

        # TODO Implement forward propogation
        # of maximum pooling module.
        b, c, h_0, w_0 = x.shape
        
        # padding section
        x_padding = []
        step = self.padding
        if step != 0:
            for image in x:
                # image.shape: channel * h * w
                d3_p = np.pad(image, step, 'constant')[step:-step]
                x_padding.append(d3_p)
            x_pad = np.array(x_padding).reshape((b, c, h_0+2*step, w_0+2*step))
        else: x_pad = x
        
        b, c, h, w = x_pad.shape
        out = np.zeros([b, c, h//self.stride, w//self.stride]) 
        self.index = np.zeros_like(x)
        for b in range(b):
            for d in range(c):
                for i in range(h//self.stride):
                    for j in range(w//self.stride):
                        _x = i *self.stride
                        _y = j *self.stride
                        out[b, d, i, j] = np.max(x[b, d, _x:_x+self.ks, _y:_y+self.ks])
                        index = np.argmax(x[b, d, _x:_x+self.ks, _y:_y+self.ks])
                        self.index[b, d, _x +index//self.ks, _y +index%self.ks ] = 1
        return out
        

        # End of todo

    def backward(self, dy):
        """Backward propagation of max pooling module.

        Args:
            dy: output delta of shape (B, C, H_out, W_out).
        Returns:
            out: input delta of shape (B, C, H_in, W_in).
        """

        # TODO Implement backward propogation
        # of maximum pooling module.
        return np.repeat(np.repeat(dy, self.stride, axis=2),self.stride, axis=3)* self.index
        # End of todo


class Dropout(Module):

    def __init__(self, p: float=0.5):

        # TODO Initialize the attributes
        # of dropout module.

        ...

        # End of todo

    def forward(self, x):

        # TODO Implement forward propogation
        # of dropout module.

        ...

        # End of todo

    def backard(self, dy):

        # TODO Implement backward propogation
        # of dropout module.

        ...

        # End of todo


if __name__ == '__main__':
    #import pdb; pdb.set_trace()
    conv = Conv2d()
    
    pass