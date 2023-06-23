## Brief
### Author - Tin Chi Pang

This folder contains my work during my GCRL2000 research placement. The functions I wrote for NLP tasks are in `nlp.py`, which you can run directly to see it in action. Various walkthrough of NLP and topic modelling can be found in the various Jupyter notebook files (`.ipynb` extension). To train a model, run the model training Jupyter notebook.

## Setting up the virtual environment
The list of modules I used can be found in the file `requirements.txt`. It is advised that you set up a **python environment** using [Anaconda](https://docs.anaconda.com/anaconda/install/index.html) (though I used [Mamba](https://github.com/conda-forge/miniforge#mambaforge) which is the C++ implementation of Anaconda and is much, much faster). Setting up an environment makes it easier to separate dependencies - modules installed in one environment are inaccessible in another. You can safely delete the environment and the modules installed in it without affecting other environments, and new environments can be quickly initialised using a `requirements.txt` file with (hopefully) no dependency errors.

To set up an Anaconda environment using `requirements.txt`, run the following:

```bash
$ conda create --name <insert_env_name_here> --file <path_to_requirements.txt>
```

Or, if you are using Mamba,

```bash
$ mamba create --name <insert_env_name_here> --file <path_to_requirements.txt>
```

---

To activate the environment you just created, run the following:
```bash
$ conda activate <insert_env_name_here>
```

Or, if you are using Mamba,
```bash
$ mamba activate <insert_env_name_here>
```

---

To deactive the environment, run the following:
```bash
$ conda deactivate
```

Or, if you are using Mamba,
```bash
$ mamba deactivate
```

## References
- Dataset obtained from 
    - https://www.kaggle.com/datasets/aneeshtickoo/tweets-on-elon-musk-twitter-acquisition
- Information on NLP obtained from
    - S. Vajjala, B. Majumder, A. Gupta, and H. Surana, “Practical Natural Language Processing [Book].” https://www.oreilly.com/library/view/practical-natural-language/9781492054047/.
    - Latent Dirichlet Allocation (Part 1 of 2), (Mar. 19, 2020). [Online Video]. Available: https://www.youtube.com/watch?v=T05t-SqKArY
