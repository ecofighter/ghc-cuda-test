module Main where

import           Control.Concurrent
import           Foreign.C.Types
import           Foreign.CUDA.Driver
import           Foreign.Ptr
import           Foreign.Storable

foreign import ccall unsafe "dot" cDot :: CInt -> IO ()

main :: IO ()
main = do
  putStrLn "will init"
  runInBoundThread $ do
    initialise []
    dev1 <- device 1
    ctx <- create dev1 []
    mdl <- loadFile "lib/gpucode.ptx"
    dotp <- getFun mdl "dotp"
    putStrLn "inited"
    let xs = [1 .. 1024] :: [CDouble]
    let ys = [2,4 .. 2048] :: [CDouble]
    xs_dev <- newListArray xs
    ys_dev <- newListArray ys
    zs_dev <- mallocArray 1024 :: IO (DevicePtr CDouble)
    launchKernel
      dotp
      (4, 1, 1)
      (256, 1, 1)
      0
      Nothing
      [VArg xs_dev, VArg ys_dev, VArg zs_dev]
    sync
    zs <- peekListArray 1024 zs_dev
    print zs
  putStrLn "end"
