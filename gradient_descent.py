# This program is to design functions for gradient descent method
# import package
import torch;
from tqdm import tqdm;

## define pearson correlation-based gradient descent 
def correlation( 
        y_pred,
        y_true
        ):
    """
    the input shape must in: ( other, sample )

    """

    # Compute deviation from mean value
    vx = y_pred - torch.nanmean( y_pred, dim=1, keepdim=True );
    vy = y_true - torch.nanmean( y_true, dim=1, keepdim=True );

   numerator   = torch.nansum( vx * vy, dim=1 );
   denominator = torch.sqrt( np.nansum( vx**2, dim=1 ) * np.nansum( vy**2, dim=1 ) + 1e-8 );
   corr = numerator / denominator;

   return -corr

## define root-mean-square error
def RMSE(
        y_pred,
        y_true
        )

    return torch.nanmean( ( y_pred - y_true )**2.0 );

def gradient_descent( 
        A_init ,       # initial guess
        X,             # Predictor
        Y,             # Predictant
        lr=1e-2,       # learning rate
        momentum=0.9,  # momentum
        n_epochs=1000, # number of epoch
        mse_rate=0.5,  # weighting on mse 
        corr_rate=0.5, # weighting on correlation,
        ):
    
    # initial weighting functions 
    A = torch.nn.Parameter( torch.from_numpy( A_init ), dtype=torch.float32 );

    # setting up optimizer 
    optimizer = torch.optim.SGD( [A], lr=lr, momentum=momentum );

    for epoch in tqdm( range( n_epochs ) ):
        optimizer.zero_grad();

        Y_pred    = A @ X;
        loss_mse  = RMSE( Y_pred, Y );
        loss_corr = correlation( Y_pred, Y );

        loss = mse_rate * loss_mse + corr_rate * loss_corr;

        loss.backward();

        optimizer.step();

    return A, Y_pred;
