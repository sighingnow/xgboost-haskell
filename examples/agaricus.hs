{-# OPTIONS_GHC -Wno-type-defaults #-}
{-# LANGUAGE ImplicitParams #-}

module Main where

import Foundation

import ML.DMLC.XGBoost

main :: IO ()
main = do
    putStrLn $ "xgboost example: agaricus"

    let ?params = [ Param "max_depth" 10
                  , Param "eta" 1
                  , Param "slient" 1
                  , Param "objective" BinaryLogistic ]
        ?debug = True

    dtrain <- xgbFromFile "examples/data/agaricus.txt.train" ?debug
    dtest <- xgbFromFile "examples/data/agaricus.txt.test" ?debug

    booster <- xgbTrain dtrain 10
    result <- xgbPredict booster dtest [] 0

    labels <- xgbGetLabel dtest

    putStrLn . show $ compareLabels labels result

    -- putStrLn . show $ valueToLabel' result

    dmatrixFree dtrain
    dmatrixFree dtest
