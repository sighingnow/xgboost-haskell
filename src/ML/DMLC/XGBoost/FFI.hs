{-# LANGUAGE GeneralizedNewtypeDeriving #-}

module ML.DMLC.XGBoost.FFI where

import Foundation
import Foundation.Array
import Foundation.Array.Internal
import Foundation.Class.Storable
import Foundation.Foreign
import Foundation.String

import qualified Prelude
import qualified Foreign.Storable as Base
import Foreign.Marshal.Alloc (alloca)

import ML.DMLC.XGBoost.Exception

{-- Types --------------------------------------------------------------------}

type StringPtr = Ptr Word8
type StringArray = Ptr StringPtr
type FloatArray = Ptr Float
type UIntArray = Ptr Word32
type DMatrixArray = Ptr DMatrix
type ByteArray = Ptr Word8

newtype DMatrix = DMatrix (Ptr ())
    deriving (Storable, Base.Storable)

newtype Booster = Booster (Ptr ())
    deriving (Storable, Base.Storable)

newtype DataIter = DataIter (Ptr ())
    deriving (Storable, Base.Storable)

newtype DataHolder = DataHolder (Ptr ())
    deriving (Storable, Base.Storable)

newtype XGBoostBatchCSR = XGBoostBatchCSR (Ptr ())
    deriving (Storable, Base.Storable)

type XGBCallbackSetData = Ptr ()
-- TODO
-- type XGBCallbackSetData =
--      FunPtr (DataHolder
--             -> XGBoostBatchCSR
--             -> IO Int)

type XGBCallbackDataIterNext = Ptr ()
-- TODO
-- type XGBCallbackDataIterNext =
--      FunPtr (DataIter
--             -> XGBCallbackSetData
--             -> DataHolderHandle
--             -> IO Int)

{-- Utilities ----------------------------------------------------------------}


boolToCInt :: Bool -> CInt
boolToCInt b = if b then 1 else 0

cintToBool :: CInt -> Bool
cintToBool i = if i == 0 then False else True

getString :: StringPtr -> IO String
getString = (fromBytesUnsafe <$>) . peekArrayEndedBy 0

withString :: String -> (StringPtr -> IO a) -> IO a
withString s = withPtr (toBytes UTF8 s)

{-- Foreign Imports ----------------------------------------------------------}

foreign import ccall unsafe "XGBGetLastError" c_xgbGetLastError
    :: IO StringPtr

foreign import ccall unsafe "XGDMatrixCreateFromFile" c_xgDMatrixCreateFromFile
    :: StringPtr
    -> CInt
    -> Ptr DMatrix
    -> IO CInt

foreign import ccall unsafe "XGDMatrixCreateFromDataIter" c_xgDMatrixCreateFromDataIter
    :: DataIter
    -> XGBCallbackDataIterNext
    -> StringPtr
    -> Ptr DMatrix
    -> IO CInt

foreign import ccall unsafe "XGDMatrixCreateFromMat" c_xgDMatrixCreateFromMat
    :: FloatArray
    -> Word64
    -> Word64
    -> CFloat
    -> Ptr DMatrix
    -> IO CInt

foreign import ccall unsafe "XGDMatrixFree" c_xgDMatrixFree
    :: DMatrix
    -> IO CInt

foreign import ccall unsafe "XGDMatrixSaveBinary" c_xgDMatrixSaveBinary
    :: DMatrix
    -> StringPtr
    -> CInt
    -> IO CInt

foreign import ccall unsafe "XGDMatrixSetFloatInfo" c_xgDMatrixSetFloatInfo
    :: DMatrix
    -> StringPtr
    -> FloatArray
    -> Word64
    -> IO CInt

foreign import ccall unsafe "XGDMatrixSetUIntInfo" c_xgDMatrixSetUIntInfo
    :: DMatrix
    -> StringPtr
    -> UIntArray
    -> Word64
    -> IO CInt

foreign import ccall unsafe "XGDMatrixGetFloatInfo" c_xgDMatrixGetFloatInfo
    :: DMatrix
    -> StringPtr
    -> Ptr Word64
    -> Ptr FloatArray
    -> IO CInt

foreign import ccall unsafe "XGDMatrixGetUIntInfo" c_xgDMatrixGetUIntInfo
    :: DMatrix
    -> StringPtr
    -> Ptr Word64
    -> Ptr UIntArray
    -> IO CInt

foreign import ccall unsafe "XGDMatrixNumRow" c_xgDMatrixNumRow
    :: DMatrix
    -> Ptr Word64
    -> IO CInt

foreign import ccall unsafe "XGDMatrixNumCol" c_xgDMatrixNumCol
    :: DMatrix
    -> Ptr Word64
    -> IO CInt

foreign import ccall unsafe "XGBoosterCreate" c_xgBoosterCreate
    :: DMatrixArray
    -> Word64
    -> Ptr Booster
    -> IO CInt

foreign import ccall unsafe "XGBoosterFree" c_xgBoosterFree
    :: Booster
    -> IO CInt

foreign import ccall unsafe "XGBoosterSetParam" c_xgBoosterSetParam
    :: Booster
    -> StringPtr
    -> FloatArray -- TODO
    -> IO CInt

foreign import ccall unsafe "XGBoosterUpdateOneIter" c_xgBoosterUpdateOneIter
    :: Booster
    -> CInt
    -> DMatrix
    -> IO CInt

foreign import ccall unsafe "XGBoosterBoostOneIter" c_xgBoosterBoostOneIter
    :: Booster
    -> DMatrix
    -> FloatArray
    -> FloatArray
    -> Word64
    -> IO CInt

foreign import ccall unsafe "XGBoosterEvalOneIter" c_xgBoosterEvalOneIter
    :: Booster
    -> CInt
    -> DMatrixArray
    -> StringArray
    -> Word64
    -> Ptr StringPtr
    -> IO CInt

foreign import ccall unsafe "XGBoosterPredict" c_xgBoosterPredict
    :: Booster
    -> DMatrix
    -> CInt
    -> CUInt
    -> Ptr Word64
    -> Ptr FloatArray
    -> IO CInt

foreign import ccall unsafe "XGBoosterLoadModel" c_xgBoosterLoadModel
    :: Booster
    -> StringPtr
    -> IO CInt

foreign import ccall unsafe "XGBoosterSaveModel" c_xgBoosterSaveModel
    :: Booster
    -> StringPtr
    -> IO CInt

foreign import ccall unsafe "XGBoosterLoadModelFromBuffer" c_xgBoosterLoadModelFromBuffer
    :: Booster
    -> ByteArray
    -> Word64
    -> IO CInt

foreign import ccall unsafe "XGBoosterGetModelRaw" c_xgBoosterGetModelRaw
    :: Booster
    -> Ptr Word64
    -> Ptr ByteArray
    -> IO CInt

foreign import ccall unsafe "XGBoosterDumpModel" c_xgBoosterDumpModel
    :: Booster
    -> StringArray
    -> CInt
    -> Ptr Word64
    -> Ptr StringArray
    -> IO CInt

foreign import ccall unsafe "XGBoosterGetAttr" c_xgBoosterGetAttr
    :: Booster
    -> StringPtr
    -> Ptr FloatArray -- TODO
    -> Ptr CInt
    -> IO CInt

foreign import ccall unsafe "XGBoosterSetAttr" c_xgBoosterSetAttr
    :: Booster
    -> StringPtr
    -> FloatArray -- TODO
    -> IO CInt

foreign import ccall unsafe "XGBoosterGetAttrNames" c_xgBoosterGetAttrNames
    :: Booster
    -> Ptr Word64
    -> Ptr StringArray
    -> IO CInt

{-- Tag Types ----------------------------------------------------------------}

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

{-- Error Handling -----------------------------------------------------------}

guard_ffi :: IO CInt -> IO ()
guard_ffi action = action >>= \r ->
    if r == 0
        then return ()
        else xgbGetLastError >>= throw . XGBError r

xgbGetLastError :: IO String
xgbGetLastError = c_xgbGetLastError >>= getString

{-- FFI Bindings -------------------------------------------------------------}

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
