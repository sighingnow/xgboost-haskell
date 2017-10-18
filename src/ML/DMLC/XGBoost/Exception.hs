module ML.DMLC.XGBoost.Exception
    ( XGBoostException (..)
    , throw
    ) where

import Foundation
import Foundation.Monad (throw)

import qualified Prelude

data XGBoostException = XGBError Int32 String deriving Typeable

instance Exception XGBoostException

instance Prelude.Show XGBoostException where
    show (XGBError err message) = "XGBError: return " <> Prelude.show err <> "\n" <> (toList message)
