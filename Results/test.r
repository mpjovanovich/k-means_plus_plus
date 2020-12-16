X <- read.table(file('stdin'),sep=' ',header=FALSE)
cat(min(X[,1]),median(X[,1]),mean(X[,1]),max(X[,1]),sd(X[,1]),
min(X[,2]),median(X[,2]),mean(X[,2]),max(X[,2]),sd(X[,2]))
