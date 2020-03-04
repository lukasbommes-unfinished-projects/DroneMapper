cd build

mkdir dist
cd dist
mkdir pyDBoW3
copy ..\Release\pyDBoW3.pyd pyDBoW3
copy c:\opencv\build\x64\vc15\bin\opencv_world400.dll pyDBoW3
copy ..\..\src\__init__.py pyDBoW3
copy ..\..\src\setup.py .
copy ..\..\src\MANIFEST.in .

python setup.py bdist_wheel

cd ..\..