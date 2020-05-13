
# Transformada inversa ----------------------------------------------------
library(tensorflow)
simulation <- function(lam, n, lrate, iter){
lambda1 <- lam

#lambda1 <- 10
f <- function(x) {((lambda1^2)+x-2*lambda1) * exp(-x/lambda1) / ((lambda1^2) * (lambda1-1))}

#curve(Fa(x), col="blue", lwd=2,add=TRUE, from=0, to=1000)

Fa <- function(x) {integrate(f,0,x)$value}
Fa <- Vectorize(Fa)

#d <- function(y){uniroot(function(x){Fa(x)-y}, interval=c(0,100))}
#d <- Vectorize(d)

#curve(d(x), col="blue", lwd=2,add=TRUE, from=0, to=1000)

#F.inv <- function(y){uniroot(function(x){Fa(x)-y},interval=c(0,1000))$root}
#F.inv <- function(y){uniroot(function(x){Fa(x)-y},interval=c(0,1), extendInt = "yes")$root}
F.inv <- function(y){uniroot(function(x){Fa(x)-y},interval=c(0,1), extendInt = "upX")$root}

F.inv <- Vectorize(F.inv)

#x <- seq(0,5,length.out=1000)
#y <- seq(0,1,length.out=1000)

#par(mfrow=c(1,3))
#plot(x,f(x),type="l",main="f(x)")
#plot(x,Fa(x),type="l",main="CDF of f(x)")
#plot(y,F.inv(y),type="l",main="Inverse CDF of f(x)")


Y <- runif(n,0,1)   # random sample from U[0,1]
Z <- F.inv(Y)

#par(mfrow=c(1,2))
#plot(x,f(x),type="l",main="Density function")
#hist(Z, breaks=20, xlim=c(0,100))

        
        #tf$random$set_random_seed(4)
        lambda <- tf$Variable(tf$random_uniform(shape(1L), 30, 60), name="lambda")
        #lambda <- tf$Variable(runif(1,75,125), name="lambda")
    
        X <- tf$placeholder(dtype=tf$float32, name = "x-data")
        
        loss <- -tf$reduce_sum(tf$log((((lambda^2)+X-2*lambda) * tf$exp(-X/lambda)) / ((lambda^2) * (lambda-1))))
        #loss <- 2*n*tf$log(lambda)+n*tf$log(lambda-1)+(1/lambda)*tf$reduce_sum(X)-tf$reduce_sum(tf$log(lambda^{2}+X-2*lambda))
        #loss <- tf$reduce_sum(-tf$log(1/(lambda^{2}*(lambda-1)))-tf$log(lambda^{2}+X-2*lambda)+X/lambda)
        
        optimizer <- tf$train$GradientDescentOptimizer(lrate)
        train <- optimizer$minimize(loss)
        
        sess = tf$Session()
        sess$run(tf$global_variables_initializer())
        
        #Create dictionary to feed the data to optimize the coefficients
        fd <- dict(X = Z)
        
        
        # Fit the line
        n_iter <- iter
        i<-0
        vecloss <- NULL
        for (step in 1:n_iter) {
                #Run session to optimize parameter
                sess$run(train, feed_dict=fd)
                #Calculate value of loss function for each iteration
                l <- sess$run(loss, feed_dict = fd)
                i <- i + 1
                vecloss[i] <- l
                #if (step %% 1000 == 0 | step <=5)
                 #       cat(step, "\t", sess$run(lambda), loss$eval(feed_dict = fd, session = sess), "\n")
                
        }
        
       
        res <- c(sess$run(lambda))
        res1 <- res
        error <- (res1-lambda1)^{2}
        bias <- (res1-lambda1)
        sess$close()
        
        return(c(error, res1, bias))
        
}

final <- replicate(500, simulation(45, 500, 0.5, 1000))
mse <- mean(final[1,])
prom <- mean(final[2,])
bias <- mean(final[3,])

write.csv(final, file="45500changenew.csv")
  #write.csv(final, file = "hola1010")
#15, 25, 64
# aceptación rechazo ------------------------------------------------------

lambda <- 3
f <- function(x) ((lambda^2)+x-2*lambda) * exp(-x/lambda) / ((lambda^2) * (lambda-1))
h <- function(x) f(x) / dexp(x, rate = lambda)
cons <- optimize(f=h, interval=c(0, 10), maximum=TRUE)$objective

curve(f, from=0, to=10, lwd=2, col=2)
curve(dexp(x, rate = lambda),
      from=0, to=10, lwd=2, col=4, add=TRUE)

curve(h)

fun <- function(n){
        
        exito <- 0
        num <- numeric(n)
        
        while(exito < n){
                y <- rexp(n=1, rate = lambda) # Value from g(x)
                u <- runif(1, 0, 1)
                if (u < f(y) / (cons * dexp(x=y, rate = lambda))){
                        exito <- exito + 1
                        num[exito] <- y
                }#end if
                
        }#end while
        
        num
}

fun(1)
