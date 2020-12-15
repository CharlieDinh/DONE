# DONE: Distributed Newton Method for Federated Edge Learning.

This repository is for the Experiment Section of the paper:
"DONE: Distributed Newton Method for Federated Edge Learning"

Link: https://arxiv.org/pdf/2012.05625.pdf

# Software requirements:
- numpy, scipy, pytorch, Pillow, matplotlib.

- To download the dependencies: **pip3 install -r requirements.txt**

- The code can be run on any pc.

# Dataset: We use 2 datasets: MNIST, and Synthetic

- To generate non-idd MNIST Data: 
  - Access data/Mnist and run: "python3 generate_niid_32users.py"
  - We can change the number of user and number of labels for each user using 2 variable NUM_USERS = 32 and NUM_LABELS = 3

- To generate non-iid Synthetic:
  - Access data/Linear_synthetic and run: "python3 generate_niid_linear_32users_updated.py". Synthetic data is configurable with the number of users, the numbers of labels for each user, and the value of $\kappa$.

- The datasets also are available to download at: https://drive.google.com/drive/folders/1LkBjkP0PzfRNiAY9ImN85r9vBIuW4U6-?usp=sharing

# Produce experiments and figures
- There is a main file "main.py" which allows running all experiments, and 2 files: "plot_mnist.py", "plot_synthetic.py" to plot all results after runing all experiment.  Only run "plot_mnist.py" and "plot_synthetic.py" after getting the results from training process.

## Effect of parameters: $\alpha$, $R $, and $\kappa$
- To produce the Fig.1 : Effects of various values of $\alpha$ and $R$ on synthetic ($\kappa = 10^2$)
<p align="center">
  <img src="https://user-images.githubusercontent.com/44039773/90308495-59b36580-df23-11ea-856e-6b1e34085715.png" height="300">
</p>
<p align="center">
  <img src="https://user-images.githubusercontent.com/44039773/90308494-58823880-df23-11ea-88c3-82a629e38894.png" height="300">
</p>
    <pre><code>
      python3 main.py --dataset Linear_synthetic --model linear_regression --algorithm DONE --batch_size 0 --alpha 0.06 --num_global_iters 100 --local_epochs 20 --numedges 32
      python3 main.py --dataset Linear_synthetic --model linear_regression --algorithm DONE --batch_size 0 --alpha 0.08 --num_global_iters 100 --local_epochs 20 --numedges 32
      python3 main.py --dataset Linear_synthetic --model linear_regression --algorithm DONE --batch_size 0 --alpha 0.1 --num_global_iters 100 --local_epochs 20 --numedges 32
      python3 main.py --dataset Linear_synthetic --model linear_regression --algorithm DONE --batch_size 0 --alpha 0.2 --num_global_iters 100 --local_epochs 20 --numedges 32
      python3 main.py --dataset Linear_synthetic --model linear_regression --algorithm DONE --batch_size 0 --alpha 0.1 --num_global_iters 100 --local_epochs 5 --numedges 32
      python3 main.py --dataset Linear_synthetic --model linear_regression --algorithm DONE --batch_size 0 --alpha 0.1 --num_global_iters 100 --local_epochs 10 --numedges 32
      python3 main.py --dataset Linear_synthetic --model linear_regression --algorithm DONE --batch_size 0 --alpha 0.1 --num_global_iters 100 --local_epochs 20 --numedges 32
      python3 main.py --dataset Linear_synthetic --model linear_regression --algorithm DONE --batch_size 0 --alpha 0.1 --num_global_iters 100 --local_epochs 30 --numedges 32
    </code></pre>
    
- To produce the Fig.2 : Effects of various values of $\alpha$ and $R$ on MNIST
<p align="center">
  <img src="https://user-images.githubusercontent.com/44039773/90308499-5d46ec80-df23-11ea-9bcc-08c748b589d7.png" height="300">
</p>
<p align="center">
  <img src="https://user-images.githubusercontent.com/44039773/90308497-5c15bf80-df23-11ea-8073-94bddbbb71f5.png" height="300">
</p>
    <pre><code>
      python3 main.py --dataset Mnist --model mclr --algorithm DONE --batch_size 0 --alpha 0.005 --num_global_iters 100 --local_epochs 40 --numedges 32
      python3 main.py --dataset Mnist --model mclr --algorithm DONE --batch_size 0 --alpha 0.01 --num_global_iters 100 --local_epochs 40 --numedges 32
      python3 main.py --dataset Mnist --model mclr --algorithm DONE --batch_size 0 --alpha 0.02 --num_global_iters 100 --local_epochs 40 --numedges 32
      python3 main.py --dataset Mnist --model mclr --algorithm DONE --batch_size 0 --alpha 0.03 --num_global_iters 100 --local_epochs 40 --numedges 32
      python3 main.py --dataset Mnist --model mclr --algorithm DONE --batch_size 0 --alpha 0.03 --num_global_iters 100 --local_epochs 10 --numedges 32
      python3 main.py --dataset Mnist --model mclr --algorithm DONE --batch_size 0 --alpha 0.03 --num_global_iters 100 --local_epochs 20 --numedges 32
      python3 main.py --dataset Mnist --model mclr --algorithm DONE --batch_size 0 --alpha 0.03 --num_global_iters 100 --local_epochs 30 --numedges 32
      python3 main.py --dataset Mnist --model mclr --algorithm DONE --batch_size 0 --alpha 0.03 --num_global_iters 100 --local_epochs 40 --numedges 32
    </code></pre>
    
## Performance comparison with different distributed algorithms
- For MNIST:
<p align="center">
  <img src="https://user-images.githubusercontent.com/44039773/90308504-61730a00-df23-11ea-88a6-4c1f43bf54c1.png" height="300">
</p>
      <pre><code>
      python3 main.py --dataset Mnist --model mclr --algorithm DONE --batch_size 0 --alpha 0.015 --num_global_iters 100 --local_epochs 120 --numedges 32
      python3 main.py --dataset Mnist --model mclr --algorithm DONE --batch_size 256 --alpha 0.01 --num_global_iters 100 --local_epochs 190 --numedges 32
      python3 main.py --dataset Mnist --model mclr --algorithm DONE --batch_size 128 --alpha 0.003 --num_global_iters 100 --local_epochs 200 --numedges 32
      python3 main.py --dataset Mnist --model mclr --algorithm Newton --batch_size 0 --alpha 0.06 --num_global_iters 100 --local_epochs 40 --numedges 32
      python3 main.py --dataset Mnist --model mclr --algorithm DANE --batch_size 0 --eta 1 --learning_rate 0.1 --num_global_iters 100 --local_epochs 40 --numedges 32
      python3 main.py --dataset Mnist --model mclr --algorithm GD --batch_size 0 --learning_rate 0.2 --num_global_iters 100 --numedges 32
    </code></pre>
- For Synthetic:
<p align="center">
  <img src="https://user-images.githubusercontent.com/44039773/90308485-51f3c100-df23-11ea-8b9e-367905260506.png" height="300">
</p>
 <pre><code>
      python3 main.py --dataset Linear_synthetic --model linear_regression --algorithm DONE --batch_size 0 --alpha 0.1 --num_global_iters 100 --local_epochs 20 --numedges 32
      python3 main.py --dataset Linear_synthetic --model linear_regression --algorithm DONE --batch_size 256 --alpha 0.1 --num_global_iters 100 --local_epochs 20 --numedges 32
      python3 main.py --dataset Linear_synthetic --model linear_regression --algorithm DONE --batch_size 128 --alpha 0.1 --num_global_iters 100 --local_epochs 20 --numedges 32
      python3 main.py --dataset Linear_synthetic --model linear_regression --algorithm Newton --batch_size 0 --alpha 0.1 --num_global_iters 100 --local_epochs 20 --numedges 32
      python3 main.py --dataset Linear_synthetic --model linear_regression --algorithm DANE --batch_size 0 --eta 1 --learning_rate 0.1 --num_global_iters 100 --local_epochs 20 --numedges 32
      python3 main.py --dataset Linear_synthetic --model linear_regression --algorithm GD --batch_size 0 --learning_rate 0.8 --num_global_iters 100 --numedges 32
</code></pre>
