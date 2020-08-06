# DONE: Distributed Newton Method for Federated Edge Learning.

This repository is for the Experiment Section of the paper:
"DONE: Distributed Newton Method for Federated Edge Learning"

# Software requirements:
- numpy, scipy, pytorch, Pillow, matplotlib.

- To download the dependencies: **pip3 install -r requirements.txt**

- The code can be run on any pc.

# Dataset: We use 2 datasets: MNIST, and Synthetic

- To generate non-idd MNIST Data: 
  - Access data/Mnist and run: "python3 generate_niid_32users.py"
  - We can change the number of user and number of labels for each user using 2 variable NUM_USERS = 32 and NUM_LABELS = 3

- To generate non-iid Synthetic:
  - Access data/Linear_synthetic and run: "python3 generate_niid_linear_32users.py". Synthetic data is configurable with the number of users, the numbers of labels for each user, and the value of $\kappa$.

- The datasets also are available to download at: https://drive.google.com/drive/folders/1LkBjkP0PzfRNiAY9ImN85r9vBIuW4U6-?usp=sharing

# Produce experiments and figures
- There is a main file "main.py" which allows running all experiments, and 2 files: plot_mnist.py, plot_synthetic.py to plot all results after runing all experiment.
## Effect of parameters: $\alpha$, $R $, and $\kappa$
- To produce the Fig.1 : Effects of various values of $\alpha$ and $R$ on synthetic ($\kappa = 10^3$)
<p align="center">
  <img src="https://github.com/CharlieDinh/DONE/files/5032598/Linear_synthetic_R_alpha_test_loss.pdf" height="300">
</p>
    <pre><code>
    python3 --dataset Linear_synthetic --model linear_regression --algorithm DONE --batch_size 0 --eta --num_global_iters 100 --local_epochs 10 --numedges 32
    </code></pre>
## Performance comparison with different  distributed algorithms
