module ML.DMLC.XGBoost.Rabit.FFI where

import Foundation
import Foundation.Array.Internal
import Foundation.Class.Storable
import Foundation.Collection
import Foundation.Foreign

import Foreign.Ptr (nullPtr)
import Foreign.Marshal.Alloc (alloca)
import qualified Foreign.Storable (peek)

import ML.DMLC.XGBoost.Foreign

foreign import ccall unsafe "RabitInit" c_rabitInit
    :: Int32
    -> StringArray
    -> IO ()

foreign import ccall unsafe "RabitFinalize" c_rabitFinalize
    :: IO ()

foreign import ccall unsafe "RabitGetRank" c_rabitGetRank
    :: IO Int32

foreign import ccall unsafe "RabitGetWorldSize" c_rabitGetWorldSize
    :: IO Int32

foreign import ccall unsafe "RabitIsDistributed" c_rabitIsDistributed
    :: IO Int32

foreign import ccall unsafe "RabitTrackerPrint" c_rabitTrackerPrint
    :: StringPtr
    -> IO ()

foreign import ccall unsafe "RabitGetProcessorName" c_rabitGetProcessorName
    :: StringPtr
    -> Ptr CULong
    -> CULong
    -> IO ()

foreign import ccall unsafe "RabitBroadcast" c_rabitBroadcast
    :: Ptr a
    -> CULong
    -> Int32
    -> IO ()

foreign import ccall unsafe "RabitAllreduce" c_rabitAllreduce
    :: Ptr a
    -> CSize
    -> Int32
    -> Int32
    -> Ptr ()
    -> Ptr ()
    -> IO ()

foreign import ccall unsafe "RabitLoadCheckPoint" c_rabitLoadCheckPoint
    :: Ptr StringPtr
    -> Ptr CULong
    -> Ptr StringPtr
    -> Ptr CULong
    -> IO Int32

foreign import ccall unsafe "RabitCheckPoint" c_rabitCheckPoint
    :: StringPtr
    -> CULong
    -> StringPtr
    -> CULong
    -> IO ()

foreign import ccall unsafe "RabitVersionNumber" c_rabitVersionNumber
    :: IO Int32

foreign import ccall unsafe "RabitLinkTag" c_rabitLinkTag
    :: IO Int32

rabitInit :: [String] -> IO ()
rabitInit args = do
    let (CountOf nlen) = length args
        argv = fromIntegral nlen
    withStringArray args $ \pargs ->
        c_rabitInit argv pargs

rabitFinalize :: IO ()
rabitFinalize = c_rabitFinalize

rabitGetrank :: IO Int32
rabitGetrank = c_rabitGetRank

rabitGetWordSize :: IO Int32
rabitGetWordSize = c_rabitGetWorldSize

rabitIsDistributed :: IO Bool
rabitIsDistributed = int32ToBool <$> c_rabitIsDistributed

rabitTrackerPrint :: String -> IO ()
rabitTrackerPrint msg = withString msg $ \pmsg -> c_rabitTrackerPrint pmsg

rabitGetProcessorName :: IO String
rabitGetProcessorName = do
    let nlimit = 64
    buf <- mutNew (CountOf nlimit)
    alloca $ \plen ->
        withMutablePtr buf $ \pbuf -> do
            c_rabitGetProcessorName pbuf plen (fromIntegral nlimit)
            nlen <- Foreign.Storable.peek plen
            getString' (CountOf (fromIntegral nlen)) pbuf

rabitBoradcast
    :: Ptr a -- ^ the pointer to send or recive buffer.
    -> Int32 -- ^ the size of data
    -> Int32 -- ^ the root of process
    -> IO ()
rabitBoradcast pdata nlen root = c_rabitBroadcast pdata (fromIntegral nlen) root

-- | Ref: https://github.com/dmlc/rabit/blob/master/include/rabit/internal/engine.h#L172
data AllreduceOpType = KMax | KMin | KSum | KBitwiseOR deriving Eq

instance Enum AllreduceOpType where
    toEnum 0 = KMax
    toEnum 1 = KMin
    toEnum 2 = KSum
    toEnum 3 = KBitwiseOR
    toEnum _ = error "No such AllreduceOpType"
    fromEnum KMax = 0
    fromEnum KMin = 2
    fromEnum KSum = 3
    fromEnum KBitwiseOR = 4

-- | Ref: https://github.com/dmlc/rabit/blob/master/include/rabit/internal/engine.h#L179
data AllreduceDataType = KChar | KUChar | KInt | KUInt | KLong | KULong | KFloat | KDouble | KLongLong | KULongLong deriving Eq

instance Enum AllreduceDataType where
    toEnum 0 = KChar
    toEnum 1 = KUChar
    toEnum 2 = KInt
    toEnum 3 = KUInt
    toEnum 4 = KLong
    toEnum 5 = KULong
    toEnum 6 = KFloat
    toEnum 7 = KDouble
    toEnum 8 = KLongLong
    toEnum 9 = KULongLong
    toEnum _ = error "No such AllreduceDataType"
    fromEnum KChar = 0
    fromEnum KUChar = 1
    fromEnum KInt = 2
    fromEnum KUInt = 3
    fromEnum KLong = 4
    fromEnum KULong = 5
    fromEnum KFloat = 6
    fromEnum KDouble = 7
    fromEnum KLongLong = 8
    fromEnum KULongLong = 9

rabitAllreduce
    :: Ptr a   -- ^ buffer for both sending and recving data
    -> Int32   -- ^ number of elements to be reduced
    -> AllreduceDataType
    -> AllreduceOpType
    -> IO ()
rabitAllreduce pdata count dtype optype =
    c_rabitAllreduce pdata
                     (fromIntegral count)
                     (fromIntegral . fromEnum $ dtype)
                     (fromIntegral . fromEnum $ optype)
                     nullPtr
                     nullPtr

rabitVersionNumber :: IO Int32
rabitVersionNumber = c_rabitVersionNumber

rabitLinkTag :: IO Int32
rabitLinkTag = c_rabitLinkTag
