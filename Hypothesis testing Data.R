# install and load readxl package
library(readxl)

# read Excel data
data <- read_excel('C:\\output3.xlsx')


# fit quadratic model to data
model <- lm(Nanoseconds ~ poly(Nodes, 2), data = data)

# fit null model to data
null_model <- lm(Nanoseconds ~ 1, data = data)

# perform ANOVA test
anova(null_model, model)

