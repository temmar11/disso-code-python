# install and load readxl package
library(readxl)

# read Excel data
data <- read_excel('C:\\Dijkstras time testing prob 0.04.xlsx')

# check the data structure and contents
str(data)
head(data)

# create a scatter plot of the data
library(ggplot2)
ggplot(data, aes(Nodes, Nanoseconds)) + geom_point()

# fit a quadratic model to the data
model <- lm(Nanoseconds ~ poly(Nodes, 2), data = data)

# check the summary of the model
summary(model)

# use the coefficients to write the quadratic equation
a <- model$coefficients[3]
b <- model$coefficients[2]
c <- model$coefficients[1]
quadratic_eq <- paste("y =", round(a, 2), "x^2 +", round(b, 2), "x +", round(c, 2))

# print the quadratic equation
print(quadratic_eq)

