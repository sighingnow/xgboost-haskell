{-# OPTIONS_GHC -Wno-orphans #-}
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
import Foundation.Collection
import Foundation.Foreign
import Foundation.Primitive
import Foundation.String

import qualified Prelude
import Foreign.Ptr (nullPtr)
import qualified Foreign.Storable as Base
import Foreign.Marshal.Alloc (alloca)
import GHC.Exts

import ML.DMLC.XGBoost.Exception

{-- Types --------------------------------------------------------------------}

type StringPtr = Ptr Word8
type StringArray = Ptr StringPtr
type FloatArray = Ptr Float
type UIntArray = Ptr Word32
type DMatrixArray = Ptr DMatrix
type ByteArray = Ptr Word8

{-- Orphan instance to make `Ptr a` as foundation's PrimType --}

instance PrimType (Ptr a) where
    primSizeInBytes _ = size (Proxy :: Proxy (Ptr a))
    {-# INLINE primSizeInBytes #-}

    primShiftToBytes _ = let (CountOf k) = size (Proxy :: Proxy (Ptr a)) in if k == 4 then 3 else 5 -- TODO may be wrong
    {-# INLINE primShiftToBytes #-}

    primBaUIndex ba (Offset (I# n)) = Ptr (indexAddrArray# ba n)
    {-# INLINE primBaUIndex #-}

    primMbaURead mba (Offset (I# n)) = primitive $ \s1 -> let !(# s2, r1 #) = readAddrArray# mba n s1
                                                           in (# s2, Ptr r1 #)
    {-# INLINE primMbaURead #-}

    primMbaUWrite mba (Offset (I# n)) (Ptr w) = primitive $ \s1 -> (# writeAddrArray# mba n w s1, () #)
    {-# INLINE primMbaUWrite #-}

    primAddrIndex addr (Offset (I# n)) = Ptr (indexAddrOffAddr# addr n)
    {-# INLINE primAddrIndex #-}

    primAddrRead addr (Offset (I# n)) = primitive $ \s1 -> let !(# s2, r1 #) = readAddrOffAddr# addr n s1
                                                            in (# s2, Ptr r1 #)
    {-# INLINE primAddrRead #-}

    primAddrWrite addr (Offset (I# n)) (Ptr w) = primitive $ \s1 -> (# writeAddrOffAddr# addr n w s1, () #)
    {-# INLINE primAddrWrite #-}

newtype DMatrix = DMatrix (Ptr ())
    deriving (Eq, Storable, Base.Storable)

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

boolToInt32 :: Bool -> Int32
boolToInt32 b = if b then 1 else 0

int32ToBool :: Int32 -> Bool
int32ToBool i = if i == 0 then False else True

getString :: StringPtr -> IO String
getString ptr
    | ptr == nullPtr = return ""
    | otherwise = fromBytesUnsafe <$> peekArrayEndedBy 0 ptr

withString :: String -> (StringPtr -> IO a) -> IO a
withString s = withPtr (toBytes UTF8 s)

getStringArray :: CountOf StringPtr -> StringArray -> IO [String]
getStringArray nlen ptr
    | ptr == nullPtr = return []
    | otherwise = do ptrs <- peekArray nlen ptr :: IO (Array StringPtr)
                     mapM getString . toList $ ptrs

getStringArray' :: StringArray -> IO [String]
getStringArray' ptr
    | ptr == nullPtr = return []
    | otherwise = do ptrs <- peekArrayEndedBy nullPtr ptr :: IO (Array StringPtr)
                     mapM getString . toList $ ptrs

withStringArray :: [String] -> (StringArray -> IO a) -> IO a
withStringArray [] f = f nullPtr
withStringArray ss f = do
    ptrs <- mapM (\s -> withString s return) ss
    withPtr (fromList ptrs) f

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
    -> FloatArray
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

{-- Tag Types ----------------------------------------------------------------}

-- | In XGBoost, the float info is correctly restricted to DMatrix's meta information, namely label and weight.
--
-- Ref: /https://github.com/dmlc/xgboost/issues/1026#issuecomment-199873890/.
data InfoField = LabelInfo | WeightInfo deriving Eq

instance Prelude.Show InfoField where
    show LabelInfo = "label"
    show WeightInfo = "weight"

-- | In XGBoost, the only uint field valid is "root_index".
--
-- Ref: /https://github.com/dmlc/xgboost/issues/1787#issuecomment-261653748/.
data UIntInfoField = RootIndex deriving Eq

instance Prelude.Show UIntInfoField where
    show RootIndex = "root_index"

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
    -> UArray Float -- ^ Value of parameter.
    -> IO ()
setParam booster name value =
    withString name $ \pname ->
        withPtr value $ \pvalue ->
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

-- | See https://github.com/dmlc/xgboost/blob/master/include/xgboost/c_api.h#L399
data PredictMask = Normal | Margin | LeafIndex | FeatureContrib

predict
    :: Booster
    -> DMatrix
    -> PredictMask
    -> Int32
    -> IO (UArray Float)
predict booster dmat mask ntree = do
    let mask' = case mask of
                    Normal -> 0
                    Margin -> 1
                    LeafIndex -> 2
                    FeatureContrib -> 4
    alloca $ \plen ->
        alloca $ \parr -> do
            guard_ffi $ c_xgBoosterPredict booster dmat mask' ntree plen parr
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

getAttr :: Booster -> String -> IO String
getAttr booster name =
    alloca $ \pout ->
        alloca $ \psucc ->
            withString name $ \pname -> do
                guard_ffi $ c_xgBoosterGetAttr booster pname pout psucc
                succ' <- peek psucc
                if int32ToBool succ'
                    then peek pout >>= getString
                    else return ""

setAttr :: Booster -> String -> String -> IO ()
setAttr booster name value =
    withString name $ \pname ->
        withString value $ \pvalue ->
            guard_ffi $ c_xgBoosterSetAttr booster pname pvalue

getAttrNames :: Booster -> IO [String]
getAttrNames booster =
    alloca $ \plen ->
        alloca $ \pout -> do
            guard_ffi $ c_xgBoosterGetAttrNames booster plen pout
            nlen <- peek plen
            peek pout >>= getStringArray (CountOf (fromIntegral nlen))
