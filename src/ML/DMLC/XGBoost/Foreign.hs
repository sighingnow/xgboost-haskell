{-# OPTIONS_GHC -Wno-orphans #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE MagicHash #-}
{-# LANGUAGE RoleAnnotations #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE UnboxedTuples #-}

module ML.DMLC.XGBoost.Foreign where

import Foundation
import Foundation.Array.Internal
import Foundation.Class.Storable
import Foundation.Collection
import Foundation.Primitive hiding (toBytes)
import Foundation.String

import Foreign.Ptr (nullPtr)
import GHC.Exts

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

{-- Types --------------------------------------------------------------------}

type StringPtr = Ptr Word8
type StringArray = Ptr StringPtr
type FloatArray = Ptr Float
type UIntArray = Ptr Word32
type ByteArray = Ptr Word8

{-- Utilities ----------------------------------------------------------------}

boolToInt32 :: Bool -> Int32
boolToInt32 b = if b then 1 else 0

int32ToBool :: Int32 -> Bool
int32ToBool i = if i == 0 then False else True

getString :: StringPtr -> IO String
getString ptr
    | ptr == nullPtr = return ""
    | otherwise = fromBytesUnsafe <$> peekArrayEndedBy 0 ptr

getString' :: CountOf Word8 -> StringPtr -> IO String
getString' nlen ptr
    | ptr == nullPtr = return ""
    | otherwise = fromBytesUnsafe <$> peekArray nlen ptr

withString :: String -> (StringPtr -> IO a) -> IO a
withString s = withPtr (toBytes UTF8 (s <> "\0"))

getStringArray :: StringArray -> IO [String]
getStringArray ptr
    | ptr == nullPtr = return []
    | otherwise = do ptrs <- peekArrayEndedBy nullPtr ptr :: IO (Array StringPtr)
                     mapM getString . toList $ ptrs

getStringArray' :: CountOf StringPtr -> StringArray -> IO [String]
getStringArray' nlen ptr
    | ptr == nullPtr = return []
    | otherwise = do ptrs <- peekArray nlen ptr :: IO (Array StringPtr)
                     mapM getString . toList $ ptrs

withStringArray :: [String] -> (StringArray -> IO a) -> IO a
withStringArray [] f = f nullPtr
withStringArray ss f = do
    ptrs <- mapM (\s -> withString s return) ss
    withPtr (fromList ptrs) f
