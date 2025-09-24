lsXX = unique(c(1,3,10,40,100))
lsNN = unique(c(1,2,5,10,40,100))

xSR <- function(data, lsxx=lsXX,lsnn=lsNN, signLS=0){
  # runRank(runSR(diff(SMA(Close,xx)),mm),nRank)
  signLS = ifelse(signLS>=0,1,-1)
  lsFactor = list(); bar=data$bar
  for(xx in lsxx){
    ma = SMA(Cl(bar), xx)
    for(nn in lsnn){
      if(nn<=1) next
      tmp = runSR(diff(ma),nn)
      tmp[is.na(tmp)]=0; tmp[tmp > 10] = 10; tmp[tmp < -10] = -10
      lsFactor[[paste("xSR",xx,nn,sep='.')]] <- tmp * signLS
    }
  }
  return(lsFactor)
}

xROC <- function(data, lsnn=lsNN, lsmm=lsNN, signLS=0){
  ## runRank(diff(Cl(bar),nn)/runSD(diff(Cl(bar)),mm),nRank)
  signLS = ifelse(signLS>=0,1,-1)
  lsFactor = list(); bar=data$bar
  for(mm in lsmm){
    if(mm<=1) next
    volLT = (runSD(diff(Cl(bar)),mm))
    for(nn in lsnn){
      if(nn>mm) next
      ratio = diff(Cl(bar),nn)/volLT
        ratio[abs(ratio)==Inf]=NA
        ratio = xts.fill2(ratio,ratio)
      lsFactor[[paste("xROC",nn,mm,sep='.')]] <- ratio * signLS
    }
  }
  return(lsFactor)
}

xdiffMA <- function(data, lsmm=lsNN, lsnn=unique(c(lsNN)), signLS=0){
  # runRank(SMA(Close,mm)/SMA(Close,nn),nRank)
  signLS = ifelse(signLS>=0,1,-1)
  lsFactor = list(); bar=data$bar
  for(mm in lsmm)
    for(nn in lsnn)
    {
      if(mm>=nn) next
      ratio = SMA(Cl(bar),mm) / SMA(Cl(bar),nn)
        ratio[abs(ratio)==Inf]=NA
        ratio = xts.fill2(ratio,ratio)
      lsFactor[[paste("xdiffMA",mm,nn,sep='.')]] <- (ratio - 1) * signLS
    }
  return(lsFactor)
}

xRSI <- function(data, lsxx=lsXX, lsnn=lsNN, signLS=0){
  ## value is symmetric
  # runRank(RSI(SMA(Close,xx),mm), nRank)
  signLS = ifelse(signLS>=0,1,-1)
  lsFactor = list(); bar=data$bar
  for(xx in lsxx){
    ma = SMA(Cl(bar),xx)
    for(nn in lsnn){
      if(nn==1) next
      lsFactor[[paste("xRSI",xx,nn,sep='.')]] <- (RSI(ma,nn)-50)/100 * signLS
    }
  }
  return(lsFactor)
}


xATR <- function(data, lsnn=lsNN){
  # runRank(myATR(bar,n),nRank)
  lsFactor = list()
  for(n in lsnn) lsFactor[[paste("xATR",n,sep='.')]] <- myATR(data$bar,n)
  return(lsFactor)
}

xSd0 <- function(data, lsxx=lsXX, lsnn=lsNN){
  ## runRank(runSD(SMA(Close,xx),nn),nRank)
  lsFactor = list()
  for(xx in lsxx){
    ma = SMA(Cl(data$bar),xx)
    for(nn in lsnn){
      if(nn<=1) next
      lsFactor[[paste("xSd0",xx,nn,sep='.')]] <- runSD(ma,nn)
    }
  }
  return(lsFactor)
}

xSd1 <- function(data, lsxx=lsXX,lsnn=lsNN){
  ## runRank(runSD(diff(SMA(Close,xx)),nn),nRank)
  lsFactor = list()
  for(xx in lsxx){
    ma = SMA(Cl(data$bar),xx)
    for(nn in lsnn){
      if(nn<=1) next
      lsFactor[[paste("xSd1",xx,nn,sep='.')]] <- runSD(diff(ma),nn)
    }
  }
  return(lsFactor)
}

xSd2 <- function(data, lsxx=lsXX,lsnn=lsNN){
  ## runRank(runSD(diff(diff(SMA(Close,xx))),nn),nRank)
  lsFactor = list()
  for(xx in lsxx){
    ma = SMA(Cl(data$bar),xx)
    for(nn in lsnn){
      if(nn<=1) next
      lsFactor[[paste("xSd2",xx,nn,sep='.')]] <- runSD(diff(diff(ma)),nn)
    }
  }
  return(lsFactor)
}

xRange <- function(data, lsnn=lsNN){
  ##runRank(myRange(bar,nn),nRank)
  lsFactor = list()
  for(nn in unique(lsnn)){
    lsFactor[[paste("xRange",nn,sep='.')]] = runMax(Hi(data$bar),nn) - runMin(Lo(data$bar),nn)
  }
  return(lsFactor)
}

xRatioRange <- function(data, lsmm=lsNN, lsnn=lsNN){
  ## runRank(myRange(bar,mm)/myRange(bar,nn), nRank)
  lsFactor = list(); bar=data$bar
  lsRange = list()
  for(kk in unique(c(lsmm,lsNN))){
    lsRange[[paste0("range",kk)]] = runMax(Hi(bar),kk) - runMin(Lo(bar),kk)
  }
  for(mm in lsmm)
    for(nn in lsnn)
    {
      if( mm >= nn) next
      range1=lsRange[[paste0("range",mm)]]; range2=lsRange[[paste0("range",nn)]]
      ratio = range1/range2
        ratio[abs(ratio)==Inf]=NA
        ratio = xts.fill2(ratio,ratio)
      lsFactor[[paste("xRatioRange",mm,nn,sep='.')]] <- ratio
    }
  return(lsFactor)
}
