module ML.DMLC.XGBoost where

import Foundation
import Foundation.Array
import Foundation.Array.Internal
import Foundation.Class.Storable
import Foundation.Collection
import Foundation.Foreign
import Foundation.Monad
import Foundation.Numerical
import Foundation.String

import qualified Prelude
import Foreign.Marshal.Alloc (alloca)

import ML.DMLC.XGBoost.FFI

data XGBoostException = XGBError CInt String deriving Typeable

instance Exception XGBoostException

instance Prelude.Show XGBoostException where
    show (XGBError err message) = "XGBError: return " <> Prelude.show err <> "\n" <> (toList message)

guard_ffi :: IO CInt -> IO ()
guard_ffi action = action >>= \r ->
    if r == 0
        then return ()
        else xgbGetLastError >>= throw . XGBError r

boolToCInt :: Bool -> CInt
boolToCInt b = if b then 1 else 0

cintToBool :: CInt -> Bool
cintToBool i = if i == 0 then False else True

xgbGetLastError :: IO String
xgbGetLastError = c_xgbGetLastError >>= getString

xgbFromFile
    :: String   -- ^ file name
    -> Bool     -- ^ print messages during loading
    -> IO DMatrix
xgbFromFile filename slient = alloca $ \pm ->
    withString filename $ \ps -> do
        guard_ffi $ c_xgDMatrixCreateFromFile ps (boolToCInt slient) pm
        peek pm

xgbFromDataIter
    :: DataIter
    -> XGBCallbackDataIterNext
    -> String
    -> IO DMatrix
xgbFromDataIter iter callback cacheinfo = alloca $ \pm ->
    withString cacheinfo $ \pc -> do
        guard_ffi $ c_xgDMatrixCreateFromDataIter iter callback pc pm
        peek pm

xgbFromMat
    :: UArray Float -- ^ mat
    -> Int          -- ^ rows
    -> Int          -- ^ columns
    -> Float        -- ^ missing value
    -> IO DMatrix
xgbFromMat arr r c missing = alloca $ \pm ->
    withPtr arr $ \parr -> do
        guard_ffi $ c_xgDMatrixCreateFromMat parr (fromIntegral r) (fromIntegral c) (CFloat missing) pm
        peek pm

dmatrixFree :: DMatrix -> IO ()
dmatrixFree = guard_ffi . c_xgDMatrixFree

-- | In XGBoost, the float info is correctly restricted to DMatrix's meta information, namely label and weight.
--
-- Ref: /https://github.com/dmlc/xgboost/issues/1026#issuecomment-199873890/.
data InfoField = Label | Weight deriving Eq

instance Prelude.Show InfoField where
    show Label = "label"
    show Weight = "weight"

-- | In XGBoost, the only uint field valid is "root_index".
--
-- Ref: /https://github.com/dmlc/xgboost/issues/1787#issuecomment-261653748/.
data UIntInfoField = RootIndex deriving Eq

instance Prelude.Show UIntInfoField where
    show RootIndex = "root_index"

xgbSetInfo
    :: DMatrix
    -> InfoField    -- ^ label field
    -> UArray Float -- ^ info vector
    -> IO ()
xgbSetInfo dm field value = do
    let (CountOf len) = length value
    withString (show field) $ \ps ->
        withPtr value $ \pv ->
            guard_ffi $ c_xgDMatrixSetFloatInfo dm ps pv (fromIntegral len)

xgbGetInfo
    :: DMatrix
    -> InfoField            -- ^ label field
    -> IO (UArray Float)    -- ^ info vector
xgbGetInfo dm field =
    alloca $ \plen ->
        alloca $ \parr ->
            withString (show field) $ \ps -> do
                guard_ffi $ c_xgDMatrixGetFloatInfo dm ps plen parr
                len <- peek plen
                arr <- peek parr
                peekArray (CountOf (fromIntegral len)) arr

xgbSetInfoUInt
    :: DMatrix
    -> UIntInfoField    -- ^ label field
    -> UArray Word32    -- ^ info vector
    -> IO ()
xgbSetInfoUInt dm field value = do
    let (CountOf len) = length value
    withString (show field) $ \ps ->
        withPtr value $ \pv ->
            guard_ffi $ c_xgDMatrixSetUIntInfo dm ps pv (fromIntegral len)

xgbGetInfoUInt
    :: DMatrix
    -> UIntInfoField        -- ^ label field
    -> IO (UArray Word32)   -- ^ info vector
xgbGetInfoUInt dm field =
    alloca $ \plen ->
        alloca $ \parr ->
            withString (show field) $ \ps -> do
                guard_ffi $ c_xgDMatrixGetUIntInfo dm ps plen parr
                len <- peek plen
                arr <- peek parr
                peekArray (CountOf (fromIntegral len)) arr


