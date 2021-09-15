##  Remove any traces from previous R-sessions
remove(ls(all=TRUE))

setwd("C:/Users/SuperUser/Documents")

##  Load the data.table package; this is convenient for data manipulation
##  If you prefer some other data manipulation package, feel free to use it
library(data.table)


## Functions
get_data_table <- function(file_name){
  return(as.data.table(read.csv(file_name)))
}


sum_by_year <- function(data_table, columns){
  return(data_table[, lapply(.SD, sum), by = Year, .SDcols = columns][order(Year)])
}


get_data_by_gender <- function(data_table){
  dt <- copy(data_table)
  male <- dt[dt$Sex == 1]
  female <- dt[dt$Sex == 2]
  return(list(Male = male, Female = female))
}



change_data_table_to_matrix <- function(data_table, column){
  mat <- as.matrix(data_table[order(Year)][ , -column, with=FALSE], rownames = data_table[order(Year)]$Year)
  Year <-  as.character(c(2000:2017))
  return(mat[Year,])
}


Calibration <- function(mu){
  Total_num_of_year <- length(YEAR)
  SVD.A <- copy(t(log(mu[,Age_over_35])))
  A_x <- rowSums(SVD.A)
  A_x <- A_x / Total_num_of_year
  SVD.A <- SVD.A - A_x
  A.SVD <- svd(SVD.A)
  B_x <- A.SVD$u[,1]
  B_x <- B_x / sum(B_x)
  K_t <- A.SVD$d[1] * A.SVD$v[,1] * sum(A.SVD$u[,1])  
  return(list(A = A_x, B = B_x, K = K_t))
}


RWD <- function (Kappa){
  diff_K <- diff(Kappa)
  theta <-mean(diff_K)
  F_YEAR <- as.character(c(2018:2060))
  TOTAL_YEAR <- as.character(c(2000:2060))
  K <- c(Kappa[length(Kappa)],seq(0, length(F_YEAR)-1))
  for (i in 1:43) {
    K[i+1] <- theta + K[i] 
  }
  TOTAL_K <- c(Kappa , K[-1])
  return(list(TOTAL = TOTAL_K, FUTURE =  K[-1] , Theta = theta))
  
}


define_as_matrix <- function(B,K,A){
  mat <- B %*% t(K) + A
  row.names(mat) <- Age_over_35
  colnames(mat) <- TOTAL_YEAR
  return(exp(t(mat)))
}


get_tot_MU <- function (MU, PRO_MU){
  i <- 1
  TOTAL_MU <- copy(MU[,Age_under_35])
  while(i <= length(F_YEAR)) {
    TOTAL_MU <- rbind(TOTAL_MU, TOTAL_MU[nrow(TOTAL_MU),])
    i <- i + 1
  }
  row.names(TOTAL_MU) <- TOTAL_YEAR
  TOTAL_MU <- cbind(TOTAL_MU, copy(PRO_MU))
  return(TOTAL_MU)
}



LIFE_EXPECTANCY <- function (MU){
  TOT_E <- rep(0,nrow(MU))
  for (j in 1:nrow(MU)) {
    M <- exp(-MU[j,])
    E <- c(M[1])
    for (i in seq(2,24)) {
      if ( (i == 2) | (i == 3) | (i == 4) | (i == 5) | (i == 6) ) {
        E <- append(E, M[i])  
      }else{
        E <- append(E,rep(M[i],5))
      }
      TOT_E[j] <- sum(cumprod(E))
    }
  }
  return(TOT_E)
}


## Load the data of number of deaths
dtICD10_1 <- get_data_table("Morticd10_part1")
dtICD10_2 <- get_data_table("Morticd10_part2")
dtICD10 <- rbind(dtICD10_1, dtICD10_2)


#remove(dtICD10_1, dtICD10_2)


## Country Code 4210 for the Netherlands and drop few columns
dtICD10 <- dtICD10[dtICD10$Country == 4210]
dtICD10 <- dtICD10[ , ":=" ( Admin1 = NULL, SubDiv = NULL, Frmat = NULL,
                             IM_Deaths1 = NULL, IM_Deaths2 = NULL, IM_Deaths3 = NULL, 
                             IM_Deaths4 = NULL, IM_Frmat = NULL, Country = NULL, Deaths26 = NULL)] 


########################################## Population ##########################################
# Load data of Expousre to the risk
pop <- get_data_table("pop")
pop <- pop[ , ":=" ( Admin1 = NULL, SubDiv = NULL, Lb = NULL, Pop26 = NULL) ] 
pop <- pop[pop$Country == 4210]

########################################## Population ##########################################
##DEFINE PUBLIC VARS
AGE <- c(0:4,seq(5,95,by = 5))
sum.cols <- grep("Death", names(dtICD10), value = T)
pop.cols <- grep("Pop", names(pop), value = T)
YEAR <-  as.character(c(2000:2017))
Age_over_35 <- paste0("Deaths", c(13:25))
Age_under_35 <- paste0("Deaths", c(2:12))
TOTAL_YEAR <- as.character(c(2000:2060))
F_YEAR <- c(2018:2060)
########################################## Plotting Mu ##################################

## By Age for Male
D <- get_data_by_gender(dtICD10)

E <- get_data_by_gender(pop)

## Drop columns and Summing by Year
D_M <- copy(D$Male)

D_M <- D_M[ , ":=" ( List = NULL, Cause = NULL, Sex = NULL)] 

D_M <- sum_by_year(D_M, sum.cols)

D_F <- copy(D$Female)

D_F <- D_F[ , ":=" ( List = NULL, Cause = NULL, Sex = NULL)] 

D_F <- sum_by_year(D_F, sum.cols)

E_M <- sum_by_year(E$Male, pop.cols)

E_F <- sum_by_year(E$Female, pop.cols)


## Getting mu
DEATH_M <- change_data_table_to_matrix(D_M, c("Year"))

DEATH_F <- change_data_table_to_matrix(D_F, c("Year"))

EXPO_M <- change_data_table_to_matrix(E_M, c("Year"))

EXPO_F <- change_data_table_to_matrix(E_F, c("Year"))

MU_M <- DEATH_M / EXPO_M

MU_F <- DEATH_F / EXPO_F


## Plot mortality rate by Age for Male
plot(AGE, MU_M["2017",sum.cols[-1]], type = "b", xlab = "Age", ylab = "log(mu)", col = 4, log = "y")
lines(AGE, MU_M["2008",sum.cols[-1]], type = "b" , col = 3)
lines(AGE, MU_M["2000",sum.cols[-1]], type = "b" , col = 1)
legend("topleft", inset = c(0,0), legend = c(2017,2008 ,2000), xpd = TRUE, col = c(4, 3 ,1), lty = 1, bty = "n", pch = 1)


## Plot mortality rate by Age for Female
plot(AGE, MU_F["2017",sum.cols[-1]], type = "b", xlab = "Age", ylab = "log(mu)", col = 4, log = "y")
lines(AGE, MU_F["2008",sum.cols[-1]], type = "b" , col = 3)
lines(AGE, MU_F["2000",sum.cols[-1]], type = "b" , col = 1)
legend("topleft", inset = c(0,0), legend = c(2017,2008 ,2000), xpd = TRUE, col = c(4, 3 ,1), lty = 1, bty = "n", pch = 1)



## All cancer deaths
All_cancer <- c(paste0("C0",0:9), paste0("C",10:97), paste0('D0',0:9), paste0('D',10:48))

DEATH_CANCER_M <- copy(D$Male[ substr(Cause,1,3) %in% All_cancer, ])

DEATH_CANCER_M <- sum_by_year(DEATH_CANCER_M, sum.cols)

DEATH_CANCER_M <- change_data_table_to_matrix(DEATH_CANCER_M,c("Year"))

MU_CANCER_M <- DEATH_CANCER_M / EXPO_M

DEATH_CANCER_F <- copy(D$Female[ substr(Cause,1,3) %in% All_cancer, ])

DEATH_CANCER_F <- sum_by_year(DEATH_CANCER_F, sum.cols)

DEATH_CANCER_F <- change_data_table_to_matrix(DEATH_CANCER_F,c("Year"))

MU_CANCER_F <- DEATH_CANCER_F / EXPO_F

par(mfrow=c(1,2))

max_ylim <- max(MU_CANCER_M[,"Deaths19"])
plot(YEAR, MU_CANCER_M[,c("Deaths19")] , type = 'b', xlab = "Year", ylab = '', main = expression(mu), ylim = c (0,max_ylim))
legend("topright", inset = c(0,0), legend = c("All Cancer"), xpd = TRUE, col = c(1), lty = 1, bty = "n", pch = 1)


## Without cancer deaths
DEATH_NO_CANCER_M <- copy(D$Male[ !substr(Cause,1,3) %in% All_cancer, ])

DEATH_NO_CANCER_M <- sum_by_year(DEATH_NO_CANCER_M, sum.cols)

DEATH_NO_CANCER_M <- change_data_table_to_matrix(DEATH_NO_CANCER_M,c("Year"))

MU_NO_CANCER_M <- DEATH_NO_CANCER_M / EXPO_M

DEATH_NO_CANCER_F <- copy(D$Female[ !substr(Cause,1,3) %in% All_cancer, ])

DEATH_NO_CANCER_F <- sum_by_year(DEATH_NO_CANCER_F, sum.cols)

DEATH_NO_CANCER_F <- change_data_table_to_matrix(DEATH_NO_CANCER_F,c("Year"))

MU_NO_CANCER_F <- DEATH_NO_CANCER_F / EXPO_F

max_ylim <- max(MU_NO_CANCER_M[,"Deaths19"])
plot(YEAR, MU_NO_CANCER_M[,c("Deaths19")] , type = 'b', xlab = "Year", ylab = '', main = expression(mu), ylim = c (0,max_ylim), col=4)
legend("topright", inset = c(0,0), legend = c("Without Cancer Deaths"), xpd = TRUE, col = c(4), lty = 1, bty = "n", pch = 1)

########################################## SVD ##########################################
## Over 35 years old
## All cause
EST_M <- Calibration(MU_M)
EST_F <- Calibration(MU_F)
EST_CANCER_M <- Calibration(MU_CANCER_M)
EST_CANCER_F <- Calibration(MU_CANCER_F)
EST_NO_CANCER_M <- Calibration(MU_NO_CANCER_M)
EST_NO_CANCER_F <- Calibration(MU_NO_CANCER_F)

NEW_AGE <- AGE[12:length(AGE)]

par(mfrow=c(3,3))

plot(NEW_AGE, EST_M$A, type = 'b', col = 4, main  = expression(alpha), xlab = '', ylab = 'ALL CAUSES')
lines(NEW_AGE, EST_F$A, type = 'b', col = 2)

plot(NEW_AGE, EST_F$B, type = 'b', col = 4, main = expression(beta), xlab = '', ylab = '')
lines(NEW_AGE, EST_M$B, type = 'b', col = 2)

plot(YEAR, EST_M$K, type = 'b', col = 4, main = expression(kappa), xlab = '', ylab = '')
lines(YEAR, EST_F$K, type = 'b', col = 2)
legend('topright',inset = c(0,-0.5), legend = c('Male', 'Female'), col = c(4,2),xpd = TRUE, horiz = TRUE, lty = 1, bty = "n", pch = 1 )

plot(NEW_AGE, EST_CANCER_M$A, type = 'b', col = 4, main  = '', xlab = '', ylab = 'ALL CANCER DEATHS')
lines(NEW_AGE, EST_CANCER_F$A, type = 'b', col = 2)

plot(NEW_AGE, EST_CANCER_F$B, type = 'b', col = 4, main = '', xlab = '', ylab = '')
lines(NEW_AGE, EST_CANCER_M$B, type = 'b', col = 2)

plot(YEAR, EST_CANCER_M$K, type = 'b', col = 4, main = '', xlab = '', ylab = '')
lines(YEAR, EST_CANCER_F$K, type = 'b', col = 2)


plot(NEW_AGE, EST_NO_CANCER_F$A, type = 'b', col = 4, main  = '', xlab = 'AGE', ylab = 'WITHOUT CANCER DEATHS')
lines(NEW_AGE, EST_NO_CANCER_M$A, type = 'b', col = 2)

plot(NEW_AGE, EST_NO_CANCER_F$B, type = 'b', col = 4, main = '', xlab = 'AGE', ylab = '')
lines(NEW_AGE, EST_NO_CANCER_M$B, type = 'b', col = 2)

plot(YEAR, EST_NO_CANCER_M$K, type = 'b', col = 4, main = '', xlab = 'YEAR', ylab = '')
lines(YEAR, EST_NO_CANCER_F$K, type = 'b', col = 2)



########################################## RWD ##########################################

## SVD
PRO_K_M <- RWD(EST_M$K)
PRO_K_F <- RWD(EST_F$K)
PRO_K_CANCER_M <- RWD(EST_CANCER_M$K)
PRO_K_CANCER_F <- RWD(EST_CANCER_F$K)
PRO_K_NO_CANCER_M <- RWD(EST_NO_CANCER_M$K)
PRO_K_NO_CANCER_F <- RWD(EST_NO_CANCER_F$K)


par(mfrow=c(1,2))

##CI is wrong (https://people.duke.edu/~rnau/411rand.htm)

plot(TOTAL_YEAR[1:18], PRO_K_M$TOTAL[1:18], main = 'MALE', xlab = 'YEAR', ylab = expression(kappa) , type ='l', col=4, ylim = c(min(PRO_K_M$TOTAL), max(PRO_K_M$TOTAL)), xlim = c(2000,2061 ))
lines(TOTAL_YEAR[18:61], PRO_K_M$TOTAL[18:61], type= 'b', col = '4')
lines(TOTAL_YEAR[1:18], PRO_K_CANCER_M$TOTAL[1:18], col = '1')
lines(TOTAL_YEAR[18:61], PRO_K_CANCER_M$TOTAL[18:61], type= 'b', col = '1')
lines(TOTAL_YEAR[1:18], PRO_K_NO_CANCER_M$TOTAL[1:18], col = '2')
lines(TOTAL_YEAR[18:61], PRO_K_NO_CANCER_M$TOTAL[18:61], type= 'b', col = '2')

plot(TOTAL_YEAR[1:18], PRO_K_F$TOTAL[1:18], main = 'FEMALE', xlab = 'YEAR', ylab = expression(kappa) , type ='l', col=4, ylim = c(min(PRO_K_M$TOTAL), max(PRO_K_M$TOTAL)), xlim = c(2000,2061 ))
lines(TOTAL_YEAR[18:61], PRO_K_F$TOTAL[18:61], type= 'b', col = '4')
lines(TOTAL_YEAR[1:18], PRO_K_CANCER_F$TOTAL[1:18], col = '1')
lines(TOTAL_YEAR[18:61], PRO_K_CANCER_F$TOTAL[18:61], type= 'b', col = '1')
lines(TOTAL_YEAR[1:18], PRO_K_NO_CANCER_F$TOTAL[1:18], col = '2')
lines(TOTAL_YEAR[18:61], PRO_K_NO_CANCER_F$TOTAL[18:61], type= 'b', col = '2')
legend('topright',inset = c(0,0), legend = c('all causes', 'all cancer', 'withtou cancer'), col = c(4,1,2),xpd = TRUE, horiz = FALSE, lty = 1, bty = "n")
legend('topright', inset = c(0,-0.1), legend = 'Projection', pch = 1, horiz = TRUE, xpd = TRUE, bty = 'n')


plot(TOTAL_YEAR[1:18], PRO_K_CANCER_F$TOTAL[1:18], main = '', xlab = '', ylab = '' , type ='l', col=1, ylim = c(min(PRO_K_CANCER_F$TOTAL), max(PRO_K_CANCER_F$TOTAL)), xlim = c(2000,2061 ))
lines(TOTAL_YEAR[18:61], PRO_K_CANCER_F$TOTAL[18:61], type= 'l', col = '3')

plot(TOTAL_YEAR[1:18], PRO_K_NO_CANCER_M$TOTAL[1:18], main ='', xlab = 'YEAR', ylab = 'WITHOUT CANCER DEATHS' , type ='l', col=4, ylim = c(min(PRO_K_NO_CANCER_M$TOTAL), max(PRO_K_NO_CANCER_M$TOTAL)), xlim = c(2000,2061 ))
lines(TOTAL_YEAR[18:61], PRO_K_NO_CANCER_M$TOTAL[18:61], type= 'l', col = '3')

plot(TOTAL_YEAR[1:18], PRO_K_NO_CANCER_F$TOTAL[1:18], main = '', xlab = 'YEAR', ylab = '' , type ='l', col=1, ylim = c(min(PRO_K_NO_CANCER_F$TOTAL), max(PRO_K_NO_CANCER_F$TOTAL)), xlim = c(2000,2061 ))
lines(TOTAL_YEAR[18:61], PRO_K_NO_CANCER_F$TOTAL[18:61], type= 'l', col = '3')





######################## PROJECTION OF MU
PRO_MU_M <- define_as_matrix(EST_M$B,PRO_K_M$TOTAL,EST_M$A)
PRO_MU_F <- define_as_matrix(EST_F$B,PRO_K_F$TOTAL,EST_F$A)
PRO_MU_CANCER_M <- define_as_matrix(EST_CANCER_M$B , PRO_K_CANCER_M$TOTAL, EST_CANCER_M$A )
PRO_MU_CANCER_F <- define_as_matrix(EST_CANCER_F$B , PRO_K_CANCER_F$TOTAL, EST_CANCER_F$A )
PRO_MU_NO_CANCER_M <- define_as_matrix(EST_NO_CANCER_M$B , PRO_K_NO_CANCER_M$TOTAL, EST_NO_CANCER_M$A )
PRO_MU_NO_CANCER_F <- define_as_matrix(EST_NO_CANCER_F$B , PRO_K_NO_CANCER_F$TOTAL, EST_NO_CANCER_F$A )


TOT_MU_M <- get_tot_MU(MU_M, PRO_MU_M)
TOT_MU_F <- get_tot_MU(MU_F, PRO_MU_F)
TOT_MU_CANCER_M <- get_tot_MU(MU_CANCER_M, PRO_MU_CANCER_M)
TOT_MU_CANCER_F <- get_tot_MU(MU_CANCER_F, PRO_MU_CANCER_F)
TOT_MU_NO_CANCER_M <- get_tot_MU(MU_NO_CANCER_M, PRO_MU_NO_CANCER_M)
TOT_MU_NO_CANCER_F <- get_tot_MU(MU_NO_CANCER_F, PRO_MU_NO_CANCER_F)

TOT_MU_MM <- TOT_MU_CANCER_M + TOT_MU_NO_CANCER_M
TOT_MU_FF <- TOT_MU_CANCER_F + TOT_MU_NO_CANCER_F

E_M <- LIFE_EXPECTANCY(TOT_MU_M)
E_F <- LIFE_EXPECTANCY(TOT_MU_F)
E_MM <- LIFE_EXPECTANCY(TOT_MU_MM)
E_FF <- LIFE_EXPECTANCY(TOT_MU_FF)
E_NO_M <- LIFE_EXPECTANCY(TOT_MU_NO_CANCER_M)
E_NO_F <- LIFE_EXPECTANCY(TOT_MU_NO_CANCER_F)



par(mfrow=c(1,2))

plot(TOTAL_YEAR, E_M, type = 'b', ylab = 'LIFE EXEPCTANCY', xlab = "YEAR", main = 'MALE')
lines(TOTAL_YEAR, E_MM , type = 'b', col=4)
lines(TOTAL_YEAR, E_NO_M , type = 'b', col=3)
legend('topright',inset = c(-0.1,-0.08), legend = c('All Cause'), col = c(1),xpd = TRUE, horiz = TRUE, lty = 1, bty = "n", pch = 1)
plot(TOTAL_YEAR, E_F, type = 'b', ylab = '', xlab = "YEAR", main = "FEMALE")
lines(TOTAL_YEAR, E_FF , type = 'b', col=4)
lines(TOTAL_YEAR, E_NO_F , type = 'b', col=3)
legend('topright',inset = c(0.1,-0.08), legend = c('Causes of death','Without Cancer'), col = c(4,3),xpd = TRUE, horiz = TRUE, lty = 1, bty = "n", pch = 1)



