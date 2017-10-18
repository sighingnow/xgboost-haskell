{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE MagicHash #-}
{-# LANGUAGE RoleAnnotations #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE UnboxedTuples #-}

module ML.DMLC.XGBoost.FFI where

import Foundation
import Foundation.Array.Internal
import Foundation.Class.Storable
import Foundation.Foreign
import Foundation.Primitive

import qualified Prelude (Show(..))
import Data.Bits ((.|.))
import qualified Foreign.Storable
import Foreign.Marshal.Alloc (alloca)
import GHC.Exts

import ML.DMLC.XGBoost.Exception
import ML.DMLC.XGBoost.Foreign

newtype DMatrix = DMatrix (Ptr ())
    deriving (Eq, Storable, Foreign.Storable.Storable)

{-- Instances to make `DMatirx` (`Ptr ()`) as foundation's PrimType, GeneralizedNewtypeDeriving doesn't work here. --}

instance PrimType DMatrix where
    primSizeInBytes _ = size (Proxy :: Proxy (Ptr ()))
    {-# INLINE primSizeInBytes #-}

    primShiftToBytes _ = let (CountOf k) = size (Proxy :: Proxy (Ptr ())) in if k == 4 then 3 else 5 -- TODO may be wrong
    {-# INLINE primShiftToBytes #-}

    primBaUIndex ba (Offset (I# n)) = DMatrix (Ptr (indexAddrArray# ba n))
    {-# INLINE primBaUIndex #-}

    primMbaURead mba (Offset (I# n)) = primitive $ \s1 -> let !(# s2, r1 #) = readAddrArray# mba n s1
                                                           in (# s2, DMatrix (Ptr r1) #)
    {-# INLINE primMbaURead #-}

    primMbaUWrite mba (Offset (I# n)) (DMatrix (Ptr w)) = primitive $ \s1 -> (# writeAddrArray# mba n w s1, () #)
    {-# INLINE primMbaUWrite #-}

    primAddrIndex addr (Offset (I# n)) = DMatrix (Ptr (indexAddrOffAddr# addr n))
    {-# INLINE primAddrIndex #-}

    primAddrRead addr (Offset (I# n)) = primitive $ \s1 -> let !(# s2, r1 #) = readAddrOffAddr# addr n s1
                                                            in (# s2, DMatrix (Ptr r1) #)
    {-# INLINE primAddrRead #-}

    primAddrWrite addr (Offset (I# n)) (DMatrix (Ptr w)) = primitive $ \s1 -> (# writeAddrOffAddr# addr n w s1, () #)
    {-# INLINE primAddrWrite #-}

type DMatrixArray = Ptr DMatrix

newtype Booster = Booster (Ptr ())
    deriving (Storable, Foreign.Storable.Storable)

newtype DataIter = DataIter (Ptr ())
    deriving (Storable, Foreign.Storable.Storable)

newtype DataHolder = DataHolder (Ptr ())
    deriving (Storable, Foreign.Storable.Storable)

newtype XGBoostBatchCSR = XGBoostBatchCSR (Ptr ())
    deriving (Storable, Foreign.Storable.Storable)

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

{-- Foreign Imports ----------------------------------------------------------}

foreign import ccall unsafe "XGBGetLastError" c_xgbGetLastError
    :: IO StringPtr

foreign import ccall unsafe "XGDMatrixCreateFromFile" c_xgDMatrixCreateFromFile
    :: StringPtr
    -> Int32
    -> Ptr DMatrix
    -> IO Int32

foreign import ccall unsafe "XGDMatrixCreateFromDataIter" c_xgDMatrixCreateFromDataIter
    :: DataIter
    -> XGBCallbackDataIterNext
    -> StringPtr
    -> Ptr DMatrix
    -> IO Int32

foreign import ccall unsafe "XGDMatrixCreateFromMat" c_xgDMatrixCreateFromMat
    :: FloatArray
    -> Word64
    -> Word64
    -> CFloat
    -> Ptr DMatrix
    -> IO Int32

foreign import ccall unsafe "XGDMatrixFree" c_xgDMatrixFree
    :: DMatrix
    -> IO Int32

foreign import ccall unsafe "XGDMatrixSaveBinary" c_xgDMatrixSaveBinary
    :: DMatrix
    -> StringPtr
    -> Int32
    -> IO Int32

foreign import ccall unsafe "XGDMatrixSetFloatInfo" c_xgDMatrixSetFloatInfo
    :: DMatrix
    -> StringPtr
    -> FloatArray
    -> Word64
    -> IO Int32

foreign import ccall unsafe "XGDMatrixSetUIntInfo" c_xgDMatrixSetUIntInfo
    :: DMatrix
    -> StringPtr
    -> UIntArray
    -> Word64
    -> IO Int32

foreign import ccall unsafe "XGDMatrixGetFloatInfo" c_xgDMatrixGetFloatInfo
    :: DMatrix
    -> StringPtr
    -> Ptr Word64
    -> Ptr FloatArray
    -> IO Int32

foreign import ccall unsafe "XGDMatrixGetUIntInfo" c_xgDMatrixGetUIntInfo
    :: DMatrix
    -> StringPtr
    -> Ptr Word64
    -> Ptr UIntArray
    -> IO Int32

foreign import ccall unsafe "XGDMatrixNumRow" c_xgDMatrixNumRow
    :: DMatrix
    -> Ptr Word64
    -> IO Int32

foreign import ccall unsafe "XGDMatrixNumCol" c_xgDMatrixNumCol
    :: DMatrix
    -> Ptr Word64
    -> IO Int32

foreign import ccall unsafe "XGBoosterCreate" c_xgBoosterCreate
    :: DMatrixArray
    -> Word64
    -> Ptr Booster
    -> IO Int32

foreign import ccall unsafe "XGBoosterFree" c_xgBoosterFree
    :: Booster
    -> IO Int32

foreign import ccall unsafe "XGBoosterSetParam" c_xgBoosterSetParam
    :: Booster
    -> StringPtr
    -> StringPtr
    -> IO Int32

foreign import ccall unsafe "XGBoosterUpdateOneIter" c_xgBoosterUpdateOneIter
    :: Booster
    -> Int32
    -> DMatrix
    -> IO Int32

foreign import ccall unsafe "XGBoosterBoostOneIter" c_xgBoosterBoostOneIter
    :: Booster
    -> DMatrix
    -> FloatArray
    -> FloatArray
    -> Word64
    -> IO Int32

foreign import ccall unsafe "XGBoosterEvalOneIter" c_xgBoosterEvalOneIter
    :: Booster
    -> Int32
    -> DMatrixArray
    -> StringArray
    -> Word64
    -> Ptr StringPtr
    -> IO Int32

foreign import ccall unsafe "XGBoosterPredict" c_xgBoosterPredict
    :: Booster
    -> DMatrix
    -> Int32
    -> Int32
    -> Ptr Word64
    -> Ptr FloatArray
    -> IO Int32

foreign import ccall unsafe "XGBoosterLoadModel" c_xgBoosterLoadModel
    :: Booster
    -> StringPtr
    -> IO Int32

foreign import ccall unsafe "XGBoosterSaveModel" c_xgBoosterSaveModel
    :: Booster
    -> StringPtr
    -> IO Int32

foreign import ccall unsafe "XGBoosterLoadModelFromBuffer" c_xgBoosterLoadModelFromBuffer
    :: Booster
    -> ByteArray
    -> Word64
    -> IO Int32

foreign import ccall unsafe "XGBoosterGetModelRaw" c_xgBoosterGetModelRaw
    :: Booster
    -> Ptr Word64
    -> Ptr ByteArray
    -> IO Int32

foreign import ccall unsafe "XGBoosterDumpModel" c_xgBoosterDumpModel
    :: Booster
    -> StringArray
    -> Int32
    -> Ptr Word64
    -> Ptr StringArray
    -> IO Int32

foreign import ccall unsafe "XGBoosterGetAttr" c_xgBoosterGetAttr
    :: Booster
    -> StringPtr
    -> Ptr StringPtr -- TODO
    -> Ptr Int32
    -> IO Int32

foreign import ccall unsafe "XGBoosterSetAttr" c_xgBoosterSetAttr
    :: Booster
    -> StringPtr
    -> StringPtr
    -> IO Int32

foreign import ccall unsafe "XGBoosterGetAttrNames" c_xgBoosterGetAttrNames
    :: Booster
    -> Ptr Word64
    -> Ptr StringArray
    -> IO Int32

foreign import ccall unsafe "XGBoosterLoadRabitCheckpoint" c_xgBoosterLoadRabitCheckpoint
    :: Booster
    -> Ptr Int32
    -> IO Int32

foreign import ccall unsafe "XGBoosterSaveRabitCheckpoint" c_xgBoosterSaveRabitCheckpoint
    :: Booster
    -> IO Int32

{-- Tag Types ----------------------------------------------------------------}

-- | In XGBoost, the float info is correctly restricted to DMatrix's meta information, namely label and weight.
--
-- Ref: /https://github.com/dmlc/xgboost/issues/1026#issuecomment-199873890/.
data FloatInfoField = LabelInfo | WeightInfo | BaseMarginInfo deriving Eq

instance Prelude.Show FloatInfoField where
    show LabelInfo = "label"
    show WeightInfo = "weight"
    show BaseMarginInfo = "base_margin"

-- | In XGBoost, the only uint field valid is "root_index".
--
-- Ref: /https://github.com/dmlc/xgboost/issues/1787#issuecomment-261653748/.
data UIntInfoField = RootIndexInfo deriving Eq

instance Prelude.Show UIntInfoField where
    show RootIndexInfo = "root_index"

-- | See https://github.com/dmlc/xgboost/blob/master/include/xgboost/c_api.h#L399
data PredictMask = Normal | Margin | LeafIndex | FeatureContrib

instance Enum PredictMask where
    toEnum 0 = Normal
    toEnum 1 = Margin
    toEnum 2 = LeafIndex
    toEnum 4 = FeatureContrib
    toEnum _ = error "No such PredictMask"
    fromEnum Normal = 0
    fromEnum Margin = 1
    fromEnum LeafIndex = 2
    fromEnum FeatureContrib = 4

{-- Error Handling -----------------------------------------------------------}

guard_ffi :: IO Int32 -> IO ()
guard_ffi action = action >>= \r ->
    if r == 0
        then return ()
        else xgbGetLastError >>= throw . XGBError r

xgbGetLastError :: IO String
xgbGetLastError = c_xgbGetLastError >>= getString

{-- DMatrix FFI Bindings -----------------------------------------------------}

xgbFromFile
    :: String   -- ^ file name
    -> Bool     -- ^ print messages during loading
    -> IO DMatrix
xgbFromFile filename slient = alloca $ \pm ->
    withString filename $ \ps -> do
        guard_ffi $ c_xgDMatrixCreateFromFile ps (boolToInt32 slient) pm
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

xgbSetFloatInfo
    :: DMatrix
    -> FloatInfoField    -- ^ label field
    -> UArray Float -- ^ info vector
    -> IO ()
xgbSetFloatInfo dm field value = do
    let (CountOf len) = length value
    withString (show field) $ \ps ->
        withPtr value $ \pv ->
            guard_ffi $ c_xgDMatrixSetFloatInfo dm ps pv (fromIntegral len)

xgbGetFloatInfo
    :: DMatrix
    -> FloatInfoField            -- ^ label field
    -> IO (UArray Float)    -- ^ info vector
xgbGetFloatInfo dm field =
    alloca $ \plen ->
        alloca $ \parr ->
            withString (show field) $ \ps -> do
                guard_ffi $ c_xgDMatrixGetFloatInfo dm ps plen parr
                len <- peek plen
                arr <- peek parr
                peekArray (CountOf (fromIntegral len)) arr

xgbGetLabel :: DMatrix -> IO (UArray Float)
xgbGetLabel mat = xgbGetFloatInfo mat LabelInfo

xgbGetWeight :: DMatrix -> IO (UArray Float)
xgbGetWeight mat = xgbGetFloatInfo mat WeightInfo

xgbGetBaseMargin :: DMatrix -> IO (UArray Float)
xgbGetBaseMargin mat = xgbGetFloatInfo mat BaseMarginInfo

xgbSetLabel :: DMatrix -> UArray Float -> IO ()
xgbSetLabel mat = xgbSetFloatInfo mat LabelInfo

xgbSetWeight :: DMatrix -> UArray Float -> IO ()
xgbSetWeight mat = xgbSetFloatInfo mat WeightInfo

xgbSetBaseMargin :: DMatrix -> UArray Float -> IO ()
xgbSetBaseMargin mat = xgbSetFloatInfo mat BaseMarginInfo

xgbSetUIntInfo
    :: DMatrix
    -> UIntInfoField    -- ^ label field
    -> UArray Word32    -- ^ info vector
    -> IO ()
xgbSetUIntInfo dm field value = do
    let (CountOf len) = length value
    withString (show field) $ \ps ->
        withPtr value $ \pv ->
            guard_ffi $ c_xgDMatrixSetUIntInfo dm ps pv (fromIntegral len)

xgbGetUIntInfo
    :: DMatrix
    -> UIntInfoField        -- ^ label field
    -> IO (UArray Word32)   -- ^ info vector
xgbGetUIntInfo dm field =
    alloca $ \plen ->
        alloca $ \parr ->
            withString (show field) $ \ps -> do
                guard_ffi $ c_xgDMatrixGetUIntInfo dm ps plen parr
                len <- peek plen
                arr <- peek parr
                peekArray (CountOf (fromIntegral len)) arr

xgbMatRow :: DMatrix -> IO Integer
xgbMatRow dm = alloca $ \pnum -> do
    guard_ffi $ c_xgDMatrixNumRow dm pnum
    fromIntegral <$> peek pnum

xgbMatCol :: DMatrix -> IO Integer
xgbMatCol dm = alloca $ \pnum -> do
    guard_ffi $ c_xgDMatrixNumCol dm pnum
    fromIntegral <$> peek pnum

{-- Booster FFI Bindings -----------------------------------------------------}

xgbBooster
    :: [DMatrix]
    -> IO Booster
xgbBooster dms = alloca $ \pb -> do
    let dms' = fromList dms
        (CountOf len) = length dms'
    withPtr dms' $ \pdm -> do
        guard_ffi $ c_xgBoosterCreate pdm (fromIntegral len) pb
        peek pb

boosterFree :: Booster -> IO ()
boosterFree = guard_ffi . c_xgBoosterFree

setParam
    :: Booster
    -> String       -- ^ Name of parameter.
    -> String       -- ^ Value of parameter.
    -> IO ()
setParam booster name value =
    withString name $ \pname ->
        withString value $ \pvalue ->
            guard_ffi $ c_xgBoosterSetParam booster pname pvalue

updateOneIter
    :: Booster
    -> Int32    -- ^ Current iteration rounds
    -> DMatrix  -- ^ Training data
    -> IO ()
updateOneIter booster iter dtrain = guard_ffi $ c_xgBoosterUpdateOneIter booster iter dtrain

boostOneIter
    :: Booster
    -> DMatrix      -- ^ Training data
    -> UArray Float -- ^ Gradient statistics
    -> UArray Float -- ^ Second order gradient statistics, should have the same length with gradient statistics array (but not checked)
    -> IO ()
boostOneIter booster dtrain grad hess = do
    let (CountOf nlen) = length grad
    withPtr grad $ \pgrad ->
        withPtr hess $ \phess ->
            guard_ffi $ c_xgBoosterBoostOneIter booster dtrain pgrad phess (fromIntegral nlen)

evalOneIter
    :: Booster
    -> Int32        -- ^ Current iteration rounds
    -> [DMatrix]    -- ^ Pointers to data to be evaluated
    -> [String]     -- ^ Names of each data, should have the same length with data array (but not checked)
    -> IO String    -- ^ The string containing evaluation statistics
evalOneIter booster iter dms names = do
    let dms' = fromList dms
        (CountOf nlen) = length dms'
    alloca $ \pstat ->
        withPtr dms' $ \pdms ->
            withStringArray names $ \pnames -> do
                guard_ffi $ c_xgBoosterEvalOneIter booster iter pdms pnames (fromIntegral nlen) pstat
                peek pstat >>= getString

boosterPredict
    :: Booster
    -> DMatrix
    -> [PredictMask]
    -> Int32
    -> IO (UArray Float)
boosterPredict booster dmat masks ntree = do
    let mask = fromIntegral $ foldl' (.|.) (fromEnum Normal) (fromEnum <$> masks)
    alloca $ \plen ->
        alloca $ \parr -> do
            guard_ffi $ c_xgBoosterPredict booster dmat mask ntree plen parr
            len <- peek plen
            arr <- peek parr
            peekArray (CountOf (fromIntegral len)) arr

loadModel
    :: Booster
    -> String   -- ^ File name
    -> IO ()
loadModel booster fname =
    withString fname $ \pfname ->
        guard_ffi $ c_xgBoosterLoadModel booster pfname

saveModel
    :: Booster
    -> String   -- ^ File name
    -> IO ()
saveModel booster fname =
    withString fname $ \pfname ->
        guard_ffi $ c_xgBoosterSaveModel booster pfname

loadModelFromBuffer
    :: Booster
    -> ByteArray   -- ^ Pointer to buffer
    -> Int32
    -> IO ()
loadModelFromBuffer booster buffer nlen = guard_ffi $ c_xgBoosterLoadModelFromBuffer booster buffer (fromIntegral nlen)

getBoosterAttr :: Booster -> String -> IO String
getBoosterAttr booster name =
    alloca $ \pout ->
        alloca $ \psucc ->
            withString name $ \pname -> do
                guard_ffi $ c_xgBoosterGetAttr booster pname pout psucc
                succ' <- peek psucc
                if int32ToBool succ'
                    then peek pout >>= getString
                    else return ""

setBoosterAttr :: Booster -> String -> String -> IO ()
setBoosterAttr booster name value =
    withString name $ \pname ->
        withString value $ \pvalue ->
            guard_ffi $ c_xgBoosterSetAttr booster pname pvalue

getAttrNames :: Booster -> IO [String]
getAttrNames booster =
    alloca $ \plen ->
        alloca $ \pout -> do
            guard_ffi $ c_xgBoosterGetAttrNames booster plen pout
            nlen <- peek plen
            peek pout >>= getStringArray' (CountOf (fromIntegral nlen))

loadRabitCheckpoint :: Booster -> IO Int32 -- ^ Return output version of the model
loadRabitCheckpoint booster =
    alloca $ \pversion -> do
        guard_ffi $ c_xgBoosterLoadRabitCheckpoint booster pversion
        peek pversion

saveRabitCheckpoint :: Booster -> IO ()
saveRabitCheckpoint booster = guard_ffi $ c_xgBoosterSaveRabitCheckpoint booster
