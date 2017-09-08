{-# LANGUAGE GeneralizedNewtypeDeriving #-}

module ML.DMLC.XGBoost.FFI where

import Foundation
import Foundation.Array
import Foundation.Array.Internal
import Foundation.Class.Storable
import Foundation.Foreign
import Foundation.String

import qualified Foreign.Storable as Base

type StringPtr = Ptr Word8
type StringArray = Ptr StringPtr
type FloatArray = Ptr Float
type UIntArray = Ptr Word32
type DMatrixArray = Ptr DMatrix
type ByteArray = Ptr Word8

getString :: StringPtr -> IO String
getString = (fromBytesUnsafe <$>) . peekArrayEndedBy 0

withString :: String -> (StringPtr -> IO a) -> IO a
withString s = withPtr (toBytes UTF8 s)

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
