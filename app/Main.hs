{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE ExtendedDefaultRules #-}

module Main where

import Foundation

import ML.DMLC.XGBoost

-- | Make float as the default type for floating literals.
default (Float)

main :: IO ()
main = do
    putStrLn $ "hello " <> "xgboost"

    dm <- xgbFromMat ([1.0, 2.0, 3.0, 4.0] :: UArray Float) 2 2 2.0

    xgbSetFloatInfo dm LabelInfo ([5.0, 6.0] :: UArray Float)
    info <- xgbGetFloatInfo dm LabelInfo
    print info

    xgbGetFloatInfo dm LabelInfo >>= print

    dmatrixFree dm

print :: Show a => a -> IO ()
print = putStrLn . show
