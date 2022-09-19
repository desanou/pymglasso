# pymglasso

Inference

## Installation

```bash
$ pip install pymglasso
```

## Usage

`from pymglasso.simulate_data import sample_block_diag_matrix`   
`p=6`   
`n=20`   
`K=3`  
`Covariance, X = sample_block_diag_matrix(n, K, p)`  



`from pymglasso.mglasso import mglasso_problem`  
`Pb = mglasso_problem(X = X, lambda1=0, lambda2_start=0.2, lambda2_factor=1.4)`


`Pb.solution`

## License

`pymglasso` was created by Edmond. It is licensed under the terms of the MIT license.

## Credits

`pymglasso` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
