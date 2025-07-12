## Deriving the ELBO (from *Temporal Difference Variational Auto-Encoder*, Gregor et al.)

We want to model the log-likelihood of the observed sequence **x**, which is:

`log p(x) = log p(x₁, x₂, ..., x_T)`

To make inference tractable, we introduce an approximate posterior distribution `q(z | x)` and use variational inference to derive a lower bound on the log-likelihood, known as the **Evidence Lower Bound (ELBO)**.

---

### Step 1: Marginal Probability

Using the definition of marginal probability, we write the predictive log-likelihood of `x_t` given past observations `x_<t` as:

`log p(x_t | x_<t) = log ∫ p(x_t, z_{t-1}, z_t | x_<t) dz_{t-1} dz_t`

---

### Step 2: Introducing the Approximate Posterior

We introduce an approximate posterior `q(z_{t-1}, z_t | x_<=t)` and rewrite the marginal using an expectation:

`log p(x_t | x_<t) = log E_{q(z_{t-1}, z_t | x_<=t)} [ p(x_t, z_{t-1}, z_t | x_<t) / q(z_{t-1}, z_t | x_<=t) ]`

---

### Step 3: Applying Jensen's Inequality

Using Jensen’s inequality, we obtain a lower bound:

`log p(x_t | x_<t) >= E_{q(z_{t-1}, z_t | x_<=t)} [ log ( p(x_t, z_{t-1}, z_t | x_<t) / q(z_{t-1}, z_t | x_<=t) ) ]`

This is the **Evidence Lower Bound (ELBO)** used for training.
