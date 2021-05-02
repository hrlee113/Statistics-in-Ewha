# ========= Setting =========

library(reshape2)
library(tidyverse)

setwd('C://Users/hyeli/Desktop/2021/금융위험관리/hw3')
source("repeated_data_management.R")

# ========= Problem1 =========

res1 = mydata[which(mydata$PolicyNum == unique(mydata$PolicyNum)[1]),][1,]
for (i in 2:length(unique(mydata$PolicyNum))){
  sam = mydata[which(mydata$PolicyNum == unique(mydata$PolicyNum)[i]),]
  row = sam[which(sam$Year==min(sam$Year)),]
  res1 = rbind(res1, row)
}
res1 = res1[,-2] # remove Year
head(res1)

res2 = as.data.frame(acast(data=mydata, PolicyNum ~ Year, value.var='n', fill=NA))
res2$PolicyNum = unique(mydata$PolicyNum)
res2 = merge(res2, data.valid[,c('PolicyNum', 'n')], by='PolicyNum', all=TRUE)
colnames(res2) = c('PolicyNum', paste0('n_', seq(2006,2011)))
head(res2)

data = merge(res2, res1, by='PolicyNum')
final_data = subset(data, select = c('PolicyNum', 'n_2006', 'n_2007', 'n_2008', 'n_2009', 'n_2010', 'n_2011',
                                     'TypeCity', 'TypeCounty', 'TypeSchool', 'TypeTown', 'TypeVillage',
                                     'col.Cov.Idx'
                                     ))

## Result
head(final_data)


# ========= Problem2 =========

library(coda)
library(rjags)
library(runjags)

X <- unname(as.matrix(final_data[8:12]))
X <- cbind(matrix(rep(1, dim(X)[1])), X)
N <- unname(as.matrix(final_data[,2:6]))
TT_N <- apply(!is.na(N), 1, sum)

# T1: indicator matrix that indicates the year of non NA N
T1 <- matrix(nrow=nrow(N), ncol=5)
for(i in 1:nrow(N)){
  ind <- which(!is.na(N[i,]))
  ind <- append(ind, rep(NA, 5-length(ind)))
  T1[i,] <- ind
}
dim(T1)
head(T1)
TTT = rep(5, dim(X)[1])

# datalist
dataList=list(N=N, X=X, TTT=TTT, TT_N=TT_N, T1=T1, 
              m.beta=c(-3, rep(0,dim(X)[2]-1)), # initial value
              inv_Sigma=1*diag(dim(X)[2]), 
              c0=.001, d0=0.001, K = dim(X)[1] # hyperparameter
)
dataList

#initial value for the parameters 
init.beta= c(-3, rep(0,dim(X)[2]-1))
init.phi = 0.8; init.sig = 0.5

# model
modelString="model {

####### Prior ########

for(i in 1:K){

    R[i,1] ~ dnorm(0, (1-phi^2)/sigsq)
    
    for(t in 2:(TTT[i]+1)){ #TTT = 5 #number of years regardless of observed or not.
        R[i,t] ~ dnorm(phi*R[i,t-1], 1/sigsq)
    }
    return_hidden_1[i] = R[i,TTT[i]+1]
}


####### Likelihood Part ####### 

for(i in 1:K){ #K: number of people
    for(t in 1:TT_N[i]){ #TT_N[i]=3: number of total observed years = 3 / 5
                       #T1[i,]= [1,3,5,NA, NA] # indication of observed years
        mu_N[i,T1[i,t]] = exp( inprod(X[i,],beta_hat[])) * exp(R[i,T1[i,t]])
        N[i,T1[i,t]] ~ dpois( mu_N[i,T1[i,t]] )
    }
}

####### prior #######
beta_hat[1:length(m.beta)] ~ dmnorm(m.beta[], inv_Sigma)
invsigsq ~ dgamma(c0,d0)
sigsq = 1/invsigsq
phi ~ dnorm(0, 1e-4)T(-1,1)

}
"
# run jags
inits1=list(beta=init.beta, sig=init.sig, 
            phi=init.phi, #sig0=init.sig, 
            .RNG.name="base::Super-Duper")
inits2=list( beta=init.beta, sig=init.sig, 
             phi=init.phi, #sig0=init.sig, 
             .RNG.name="base::Wichmann-Hill")
inits3=list(beta=init.beta, sig=init.sig, 
            phi=init.phi, #sig0=init.sig, 
            .RNG.name="base::Mesenne-Twister")

nChains=3; nAdapt=5000; nUpdate=30000; nSamples=30000; nthin=5
ptm.init <- proc.time()
runJagsOut = run.jags(method="parallel", model=modelString, 
                      monitor=c("beta_hat", "sigsq", "phi"),
                      data=dataList, inits=list(inits1, inits2, inits3),
                      n.chains=nChains, adapt=nAdapt, burnin=nUpdate,
                      sample=ceiling(nSamples/nChains), thin=nthin,
                      summarise=TRUE, plots=TRUE)
summary(runJagsOut)[,c(1,3)]


# ========= Problem3 =========
post_beta = summary(runJagsOut)[1:6,2]
post_sigsq = summary(runJagsOut)[7,2]
post_phi = summary(runJagsOut)[8,2]

# datalist
dataList=list(N=N, X=X, TTT=TTT, TT_N=TT_N, T1=T1, 
              beta=post_beta, sigsq=post_sigsq, 
              phi=post_phi, K = dim(X)[1] # hyperparameter
)

# model
modelString="model {

####### Prior ########

for(i in 1:K){

    R[i,1] ~ dnorm(0, (1-phi^2)/sigsq)
    
    for(t in 2:(TTT[i]+1)){ #TTT = 5 #number of years regardless of observed or not.
        R[i,t] ~ dnorm(phi*R[i,t-1], 1/sigsq)
    }
    return_hidden_1[i] = R[i,TTT[i]+1] # R[i,t+1]
}


####### Likelihood Part ####### 

for(i in 1:K){ #K: number of people
    for(t in 1:TT_N[i]){ #TT_N[i]=3: number of total observed years = 3 / 5
                       #T1[i,]= [1,3,5,NA, NA] # indication of observed years
        mu_N[i,T1[i,t]] = exp( X[i,] %*% beta[]) * exp(R[i,T1[i,t]])
        N[i,T1[i,t]] ~ dpois( mu_N[i,T1[i,t]] )
    }
    return_mu_N[i] = mu_N[i,T1[i,TT_N[i]]]
}
}
"

# run jags
inits1=list(.RNG.name="base::Super-Duper")
inits2=list(.RNG.name="base::Wichmann-Hill")
inits3=list(.RNG.name="base::Mesenne-Twister")
nChains=3; nAdapt=5000; nUpdate=30000; nSamples=30000; nthin=5
ptm.init <- proc.time()
runJagsOut1 = run.jags(method="parallel", model=modelString, 
                      monitor=c("return_hidden_1", "return_mu_N"),
                      data=dataList, 
                      n.chains=nChains, adapt=nAdapt, burnin=nUpdate,
                      sample=ceiling(nSamples/nChains), thin=nthin,
                      summarise=TRUE, plots=TRUE)

pred_R = do.call(rbind.data.frame, as.mcmc.list(runJagsOut1, 'return_hidden_1')) # R[i,t+1]
pred_R[,407] # 1st person, 
pred_mu_N = do.call(rbind.data.frame, as.mcmc.list(runJagsOut1, 'return_mu_N')) # lambda[i,t+1]
head(pred_mu_N[,1]) # 1st person

K = dim(pred_R)[1] # num of simulation
J = dim(pred_R)[2] # num of people

pred_N1 = rep(0, J)
for (i in 1:J){
  res = exp(pred_R[,i]) * pred_mu_N[,i]
  pred_N1[i] = sum(res) / K
}

idx = unique(mydata$PolicyNum) %in% data.valid$PolicyNum
mse = sum((data.valid$n - pred_N1[idx])^2 / J)
mae = sum(abs(data.valid$n - pred_N1[idx]) / J)
c(mse, mae)

# ========= Problem4 =========

mycov <- function(t1, t2, lamb1, lamb2){
  if (t1==t2){
    var_R = post_sigsq + post_phi^2 * post_sigsq / (1 - post_phi^2)
    return(lamb1*lamb2*var_R)
  }else{
    var_R = post_phi * post_sigsq / (1 - post_phi^2)
    return(lamb1*lamb2*var_R)
  }
}

covMat <- function(t_vec, lamb_vec){ #t_vec = n
  t_vec = na.omit(t_vec)
  t_vec = t_vec[ifelse(t_vec!=0,T,F)]
  t = length(t_vec)
  res = matrix(rep(0, t^2), nrow=t)
  for (i in 1:t){
    for (j in 1:t){
      res[i,j] = mycov(i, j, lamb_vec[i], lamb_vec[j])
    }
  }
  return(res)
}

rownames(data.valid) = NULL 
X_new <- merge(final_data[,c(1,2)], data.valid[,c(1,9,10,12,13,14)], key='PolicyNum', all.x=TRUE)
X_new <- X_new[,-c(1,2)]
X_new <- cbind(rep(1, dim(X_new)[1]), X_new)
myindex <- unique(mydata$PolicyNum) %in% data.valid$PolicyNum
n_new = sum(myindex)

pred_lambda2

# lambda에 들어갈 random effect
R <- matrix(rep(0, J*6), nrow=J)
for (j in 1:J){
  R[j,1] = dnorm(0, post_sigsq/(1-post_phi^2))
  for (n in 2:6){
    R[j,n] = dnorm(post_phi*R[j,n-1], post_phi*post_sigsq/(1-post_phi^2))
  }
}

# 시간별로 다른 lambda
for (j in 1:J){
  lambda[j]
}
lambda <- exp(R)[,1:5] * c(exp(as.matrix(X) %*% as.matrix(post_beta))) 
lambda
pred_lambda2 = exp(as.matrix(X_new) %*% as.matrix(post_beta)) * exp(R)[,6]
pred_lambda2

bp <- rep(0, J)
for (j in 1:J){
  d = ifelse(N[j,]==0, NA, N[j,])
  idx = !is.na(d)
  idx_n = sum(idx)
  if (idx_n > 0){
    C <- matrix(rep(0, idx_n))
    for (t in 1:idx_n){
      var_R = post_phi * post_sigsq / (1 - post_phi^2)
      C[t] = lambda[j,t]*pred_lambda2[j]*var_R
    }
    V <- covMat(N[j,], lambda[j,])
    A <- solve(V) %*% C
    A0 <- pred_lambda2[j] - sum(A * lambda[j,][idx])
    AA <- c(A0, A)
    bp[j] <- c(1, N[j,][idx]) %*% AA
  }
}
bp
idx = unique(mydata$PolicyNum) %in% data.valid$PolicyNum
mse = sum((data.valid$n - bp[idx])^2 / J)
mae = sum(abs(data.valid$n - bp[idx]) / J)
c(mse, mae)

# ========= Problem5 =========

# Bootsampling
n = 10000
X_boot = matrix(rep(0, n*dim(X)[2]), nrow=10000)
N_boot = matrix(rep(0, n*(dim(N)[2]+1)), nrow=10000)
R_boot = matrix(rep(0, n*6), nrow=10000)
X_new_boot = matrix(rep(0, n*dim(X_new)[2]), nrow=10000)

idx = sample(seq(1,dim(X)[1]), n, replace=T)
for (i in 1:n){
  idxx = idx[i]
  X_boot[i,] = as.matrix(X[idxx,]); N_boot[i,] = as.matrix(final_data[idxx,2:7])
  R_boot[i,] = as.matrix(R[idxx,]); X_new_boot[i,] = as.matrix(X_new[idxx,])
}
lambda = exp(R_boot)[,1:5] * c(exp(as.matrix(X_boot) %*% as.matrix(post_beta)))
pred_lambda2 = exp(as.matrix(X_new_boot) %*% as.matrix(post_beta)) * exp(R_boot)[,6]

bp <- rep(0, J)
for (j in 1:J){
  d = ifelse(N_boot[j,1:5]==0, NA, N_boot[j,1:5])
  idx = !is.na(d)
  idx_n = sum(idx)
  if (idx_n > 0){
    C <- matrix(rep(0, idx_n))
    for (t in 1:idx_n){
      var_R = post_phi * post_sigsq / (1 - post_phi^2)
      C[t] = lambda[j,t]*pred_lambda2[j]*var_R
    }
    W <- as.matrix((1/(X_boot[j,]^2*pred_lambda2[j,])*X_boot[j,]*pred_lambda2[j,])^2)[which(idx==T),]
    na.idx = !is.na(W)
    for (w in 1:length(W)){
      W[w] = ifelse(is.na(W[w]), 0, W[w])
    }
    M <- covMat(N_boot[j,1:5], lambda[j,])
    V <- M * W
    V <- V[na.idx, na.idx]
    if (sum(na.idx)>0){
      A <- solve(V) %*% C[na.idx]
      l <- lambda[j,][idx]
      A0 <- pred_lambda2[j] - sum(A * l[na.idx])
      AA <- c(A0, A)
      n <- N_boot[j,1:5][idx]
      bp[j] <- c(1, n[na.idx]) %*% AA
    }
  }
}
bp
idx = unique(mydata$PolicyNum) %in% data.valid$PolicyNum
mse = sum((data.valid$n - bp[idx])^2 / J)
mae = sum(abs(data.valid$n - bp[idx]) / J)
c(mse, mae)

# ========= Problem6 =========
# ========= Setting =========

library(reshape2)
library(tidyverse)

setwd('C://Users/hyeli/Desktop/2021/금융위험관리/hw3')
source("repeated_data_management.R")

# 1

res1 = mydata[which(mydata$PolicyNum == unique(mydata$PolicyNum)[1]),][1,]
for (i in 2:length(unique(mydata$PolicyNum))){
  sam = mydata[which(mydata$PolicyNum == unique(mydata$PolicyNum)[i]),]
  row = sam[which(sam$Year==min(sam$Year)),]
  res1 = rbind(res1, row)
}
res1 = res1[,-2] # remove Year
head(res1)

res2 = as.data.frame(acast(data=mydata, PolicyNum ~ Year, value.var='n', fill=NA))
res2$PolicyNum = unique(mydata$PolicyNum)
res2 = merge(res2, data.valid[,c('PolicyNum', 'n')], by='PolicyNum', all=TRUE)
colnames(res2) = c('PolicyNum', paste0('n_', seq(2006,2011)))
head(res2)

data = merge(res2, res1, by='PolicyNum')
final_data = subset(data, select = c('PolicyNum', 'n_2006', 'n_2007', 'n_2008', 'n_2009', 'n_2010', 'n_2011',
                                     'TypeCity', 'TypeCounty', 'TypeSchool', 'TypeTown', 'TypeVillage',
                                     'col.Cov.Idx'
))

## Result
head(final_data)


# 2

library(coda)
library(rjags)
library(runjags)

X <- unname(as.matrix(final_data[8:12]))
X <- cbind(matrix(rep(1, dim(X)[1])), X)
N <- unname(as.matrix(final_data[,2:6]))
TT_N <- apply(!is.na(N), 1, sum)

# T1: indicator matrix that indicates the year of non NA N
T1 <- matrix(nrow=nrow(N), ncol=5)
for(i in 1:nrow(N)){
  ind <- which(!is.na(N[i,]))
  ind <- append(ind, rep(NA, 5-length(ind)))
  T1[i,] <- ind
}
dim(T1)
head(T1)
TTT = rep(5, dim(X)[1])

# datalist
dataList=list(N=N, X=X, TTT=TTT, TT_N=TT_N, T1=T1, 
              m.beta=c(-3, rep(0,dim(X)[2]-1)), # initial value
              inv_Sigma=1*diag(dim(X)[2]), 
              K = dim(X)[1], r1 = 3, r2=5, p=0.3 # hyperparameter
)
dataList

#initial value for the parameters 
init.beta= c(-3, rep(0,dim(X)[2]-1))
init.phi = 0.8; init.sig = 0.5

# model
modelString="model {

####### Prior ########

for(i in 1:K){

    R[i,1] ~ dgamma(r1, r2)
    
    for(t in 2:(TTT[i]+1)){ #TTT = 5 #number of years regardless of observed or not.
        R[i,t] ~ dgamma(bigbeta*r1, r2)
    }
    return_hidden_1[i] = R[i,TTT[i]+1]
}


####### Likelihood Part ####### 

for(i in 1:K){ #K: number of people
    for(t in 1:TT_N[i]){ #TT_N[i]=3: number of total observed years = 3 / 5
                       #T1[i,]= [1,3,5,NA, NA] # indication of observed years
        mu_N[i,T1[i,t]] = exp( inprod(X[i,],beta_hat[])) * exp(R[i,T1[i,t]])
        N[i,T1[i,t]] ~ dpois( mu_N[i,T1[i,t]] )
    }
}

####### prior #######
bigbeta ~ dbeta(r1*p, r2*(1-p))
beta_hat[1:length(m.beta)] ~ dmnorm(m.beta[], inv_Sigma)
}
"
# run jags
inits1=list(beta_hat=init.beta,.RNG.name="base::Super-Duper")
inits2=list(beta_hat=init.beta,.RNG.name="base::Wichmann-Hill")
inits3=list(beta_hat=init.beta,.RNG.name="base::Mesenne-Twister")

nChains=3; nAdapt=5000; nUpdate=30000; nSamples=30000; nthin=5
ptm.init <- proc.time()
runJagsOut = run.jags(method="parallel", model=modelString, 
                      monitor=c("beta_hat", "bigbeta"),
                      data=dataList, inits=list(inits1, inits2, inits3),
                      n.chains=nChains, adapt=nAdapt, burnin=nUpdate,
                      sample=ceiling(nSamples/nChains), thin=nthin,
                      summarise=TRUE, plots=TRUE)
summary(runJagsOut)[,c(1,3)]


# 3
post_beta = summary(runJagsOut)[1:6,2]
post_bigbeta = summary(runJagsOut)[7,2]

# datalist
dataList=list(N=N, X=X, TTT=TTT, TT_N=TT_N, T1=T1, 
              post_beta=post_beta, r1 = 3, r2=5,
              post_bigbeta=post_bigbeta, K = dim(X)[1] # hyperparameter
)

# model
modelString="model {

####### Prior ########

for(i in 1:K){

    R[i,1] ~ dgamma(r1, r2)
    
    for(t in 2:(TTT[i]+1)){ #TTT = 5 #number of years regardless of observed or not.
        R[i,t] ~ dgamma(post_bigbeta*r1, r2)
    }
    return_hidden_1[i] = R[i,TTT[i]+1]
}


####### Likelihood Part ####### 

for(i in 1:K){ #K: number of people
    for(t in 1:TT_N[i]){ #TT_N[i]=3: number of total observed years = 3 / 5
                       #T1[i,]= [1,3,5,NA, NA] # indication of observed years
        mu_N[i,T1[i,t]] = exp( inprod(X[i,],post_beta[])) * exp(R[i,T1[i,t]])
        N[i,T1[i,t]] ~ dpois( mu_N[i,T1[i,t]] )
    }
    return_mu_N[i] = mu_N[i,T1[i,TT_N[i]]]
}
}
"

# run jags
inits1=list(.RNG.name="base::Super-Duper")
inits2=list(.RNG.name="base::Wichmann-Hill")
inits3=list(.RNG.name="base::Mesenne-Twister")
nChains=3; nAdapt=5000; nUpdate=30000; nSamples=30000; nthin=5
ptm.init <- proc.time()
runJagsOut1 = run.jags(method="parallel", model=modelString, 
                       monitor=c("return_hidden_1", "return_mu_N"),
                       data=dataList, 
                       n.chains=nChains, adapt=nAdapt, burnin=nUpdate,
                       sample=ceiling(nSamples/nChains), thin=nthin,
                       summarise=TRUE, plots=TRUE)

pred_R = do.call(rbind.data.frame, as.mcmc.list(runJagsOut1, 'return_hidden_1')) # R[i,t+1]
pred_R[,407] # 1st person, 
pred_mu_N = do.call(rbind.data.frame, as.mcmc.list(runJagsOut1, 'return_mu_N')) # lambda[i,t+1]
head(pred_mu_N[,1]) # 1st person

K = dim(pred_R)[1] # num of simulation
J = dim(pred_R)[2] # num of people

pred_N1 = rep(0, J)
for (i in 1:J){
  res = exp(pred_R[,i]) * pred_mu_N[,i]
  pred_N1[i] = sum(res) / K
}

idx = unique(mydata$PolicyNum) %in% data.valid$PolicyNum
mse = sum((data.valid$n - pred_N1[idx])^2 / J)
mae = sum(abs(data.valid$n - pred_N1[idx]) / J)
c(mse, mae)

# 4

mycov <- function(t1, t2, lamb1, lamb2){
  if (t1==t2){
    var_R = post_sigsq + post_phi^2 * post_sigsq / (1 - post_phi^2)
    return(lamb1*lamb2*var_R)
  }else{
    var_R = post_phi * post_sigsq / (1 - post_phi^2)
    return(lamb1*lamb2*var_R)
  }
}

covMat <- function(t_vec, lamb_vec){ #t_vec = n
  t_vec = na.omit(t_vec)
  t_vec = t_vec[ifelse(t_vec!=0,T,F)]
  t = length(t_vec)
  res = matrix(rep(0, t^2), nrow=t)
  for (i in 1:t){
    for (j in 1:t){
      res[i,j] = mycov(i, j, lamb_vec[i], lamb_vec[j])
    }
  }
  return(res)
}

rownames(data.valid) = NULL 
X_new <- merge(final_data[,c(1,2)], data.valid[,c(1,9,10,12,13,14)], key='PolicyNum', all.x=TRUE)
X_new <- X_new[,-c(1,2)]
X_new <- cbind(rep(1, dim(X_new)[1]), X_new)
myindex <- unique(mydata$PolicyNum) %in% data.valid$PolicyNum
n_new = sum(myindex)

# lambda에 들어갈 random effect
# lambda에 들어갈 random effect
r1 = 3; r2=5
R <- matrix(rep(0, J*6), nrow=J)
for (j in 1:J){
  R[j,1] = dgamma(r1, r2)
  for (n in 2:6){
    R[j,n] = dgamma(post_bigbeta*r1, r2)
  }
}


# 시간별로 다른 lambda
lambda <- exp(R)[,1:5] * c(exp(as.matrix(X) %*% as.matrix(post_beta))) 
lambda
pred_lambda2 = exp(as.matrix(X_new) %*% as.matrix(post_beta)) * exp(R)[,6]
pred_lambda2

bp <- rep(0, J)
for (j in 1:J){
  d = ifelse(N[j,]==0, NA, N[j,])
  idx = !is.na(d)
  idx_n = sum(idx)
  if (idx_n > 0){
    C <- matrix(rep(0, idx_n))
    for (t in 1:idx_n){
      var_R = post_phi * post_sigsq / (1 - post_phi^2)
      C[t] = lambda[j,t]*pred_lambda2[j]*var_R
    }
    V <- covMat(N[j,], lambda[j,])
    A <- solve(V) %*% C
    A0 <- pred_lambda2[j] - sum(A * lambda[j,][idx])
    AA <- c(A0, A)
    bp[j] <- c(1, N[j,][idx]) %*% AA
  }
}
bp
idx = unique(mydata$PolicyNum) %in% data.valid$PolicyNum
mse = sum((data.valid$n - bp[idx])^2 / J)
mae = sum(abs(data.valid$n - bp[idx]) / J)
c(mse, mae)

# 5

# Bootsampling
n = 10000
X_boot = matrix(rep(0, n*dim(X)[2]), nrow=10000)
N_boot = matrix(rep(0, n*(dim(N)[2]+1)), nrow=10000)
R_boot = matrix(rep(0, n*6), nrow=10000)
X_new_boot = matrix(rep(0, n*dim(X_new)[2]), nrow=10000)

idx = sample(seq(1,dim(X)[1]), n, replace=T)
for (i in 1:n){
  idxx = idx[i]
  X_boot[i,] = as.matrix(X[idxx,]); N_boot[i,] = as.matrix(final_data[idxx,2:7])
  R_boot[i,] = as.matrix(R[idxx,]); X_new_boot[i,] = as.matrix(X_new[idxx,])
}
lambda = exp(R_boot)[,1:5] * c(exp(as.matrix(X_boot) %*% as.matrix(post_beta)))
pred_lambda2 = exp(as.matrix(X_new_boot) %*% as.matrix(post_beta)) * exp(R_boot)[,6]

bp <- rep(0, J)
for (j in 1:J){
  d = ifelse(N_boot[j,1:5]==0, NA, N_boot[j,1:5])
  idx = !is.na(d)
  idx_n = sum(idx)
  if (idx_n > 0){
    C <- matrix(rep(0, idx_n))
    for (t in 1:idx_n){
      var_R = post_phi * post_sigsq / (1 - post_phi^2)
      C[t] = lambda[j,t]*pred_lambda2[j]*var_R
    }
    W <- as.matrix((1/(X_boot[j,]^2*pred_lambda2[j,])*X_boot[j,]*pred_lambda2[j,])^2)[which(idx==T),]
    na.idx = !is.na(W)
    for (w in 1:length(W)){
      W[w] = ifelse(is.na(W[w]), 0, W[w])
    }
    M <- covMat(N_boot[j,1:5], lambda[j,])
    V <- M * W
    V <- V[na.idx, na.idx]
    if (sum(na.idx)>0){
      A <- solve(V) %*% C[na.idx]
      l <- lambda[j,][idx]
      A0 <- pred_lambda2[j] - sum(A * l[na.idx])
      AA <- c(A0, A)
      n <- N_boot[j,1:5][idx]
      bp[j] <- c(1, n[na.idx]) %*% AA
    }
  }
}
bp
idx = unique(mydata$PolicyNum) %in% data.valid$PolicyNum
mse = sum((data.valid$n - bp[idx])^2 / J)
mae = sum(abs(data.valid$n - bp[idx]) / J)
c(mse, mae)
