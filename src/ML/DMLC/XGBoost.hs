{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE ImplicitParams #-}
{-# LANGUAGE RecordWildCards #-}

module ML.DMLC.XGBoost
    ( module ML.DMLC.XGBoost
    , module ML.DMLC.XGBoost.FFI
    ) where

import Foundation
import Foundation.Collection
import Foundation.Numerical

import qualified Prelude (fromIntegral, Show(..))
import Control.Exception (assert)
import Control.Monad (when, foldM_)

import ML.DMLC.XGBoost.FFI
import ML.DMLC.XGBoost.Rabit.FFI

{------------------------------------------------------------------------------
-- Utility functions.
------------------------------------------------------------------------------}

integerToFloat :: Int -> Float
integerToFloat = Prelude.fromIntegral

-- | Cast floating point output to integer label.
valueToLabel :: (IntegralRounding a) => a -> Int32
valueToLabel = roundNearest

{-# SPECIALIZE valueToLabel :: Float -> Int32 #-}

{-# INLINE valueToLabel #-}

compareLabels
    :: UArray Float -- ^ Wanted
    -> UArray Float -- ^ Actual output
    -> Float        -- ^ Successfully rate
compareLabels wanted actual = integerToFloat sameLength / integerToFloat nLength
    where
        sameLabel a b = valueToLabel a == valueToLabel b
        accLabels (a, b) n = if sameLabel a b
                                then n + 1
                                else n
        nLength = let (CountOf k) = min (length wanted) (length actual) in k
        sameLength = foldr' accLabels 0 $ (zipWith (,) wanted actual :: [(Float, Float)])

compareLabels'
    :: UArray Float             -- ^ Wanted
    -> UArray Float             -- ^ Actual output
    -> (Float -> Float -> Bool) -- ^ decide whether two given labels are equal
    -> Float                    -- ^ Successfully rate
compareLabels' wanted actual sameLabel = integerToFloat sameLength / integerToFloat nLength
    where
        accLabels (a, b) n = if sameLabel a b
                                then n + 1
                                else n
        nLength = let (CountOf k) = min (length wanted) (length actual) in k
        sameLength = foldr' accLabels 0 $ (zipWith (,) wanted actual :: [(Float, Float)])


{------------------------------------------------------------------------------
-- DMatrix related APIs.
------------------------------------------------------------------------------}



{------------------------------------------------------------------------------
-- Booster related APIs.
------------------------------------------------------------------------------}

-- | Parameter passed to booster.
data BoosterParam = forall a. Show a => Param { paramName :: String
                                              , paramValue :: a
                                              }

-- | Predefined objective functions.
--
-- Ref: https://github.com/dmlc/xgboost/blob/master/src/objective/regression_obj.cc
data ObjectiveFunction = RegLinear | RegLogistic | BinaryLogistic | BinaryLogitraw | CountPoisson | RegGamma | RegTweedie deriving Eq

instance Show ObjectiveFunction where
    show RegLinear = "reg:linear"
    show RegLogistic = "reg:logistic"
    show BinaryLogistic = "binary:logistic"
    show BinaryLogitraw = "binary:logitraw"
    show CountPoisson = "count:poisson"
    show RegGamma = "reg:gamma"
    show RegTweedie = "reg:tweedie"

setBoosterParam
    :: Booster
    -> BoosterParam
    -> IO ()
setBoosterParam booster Param{..} = setParam booster paramName (show paramValue)

newBooster :: (?params :: [BoosterParam]) => [DMatrix] -> IO Booster
newBooster dmats = do
    booster <- xgbBooster dmats
    setParam booster "seed" "0"
    forM_ ?params $ \param ->
        setBoosterParam booster param
    return booster

{------------------------------------------------------------------------------
-- Model train and predict APIs.
------------------------------------------------------------------------------}

xgbTrain
    :: (?params :: [BoosterParam], ?debug :: Bool)
    => DMatrix  -- ^ Data to be trained
    -> Int32    -- ^ Number of boosting iterations
    -> IO Booster
xgbTrain dtrain rounds = do
    let nboost = 0

    booster <- newBooster [dtrain]
    version <- loadRabitCheckpoint booster

    when ?debug $ do
        wdsize <- rabitGetWordSize
        assert (wdsize /= 1 || version == 0) $ return ()

    let startIter = version `div` 2

    let go (_nboost, _version) i = do
            _version' <- if _version `mod` 2 == 0
                            then do
                                updateOneIter booster i dtrain
                                saveRabitCheckpoint booster
                                return (_version + 1)
                            else return _version

            when ?debug $ do
                wdsize <- rabitGetWordSize
                ver <- rabitVersionNumber
                assert (wdsize == 1 || _version' == ver) $ return ()

            saveRabitCheckpoint booster

            return (_nboost + 1, _version' + 1)

    foldM_ go (nboost + startIter, version) [startIter, rounds]

    return booster

xgbPredict
    :: (?debug :: Bool)
    => Booster
    -> DMatrix
    -> [PredictMask]
    -> Int32            -- ^ Limit number of trees in the prediction; defaults to 0 (use all trees).
    -> Bool             -- ^ `training` parameter
    -> IO (UArray Float)
xgbPredict = boosterPredict
