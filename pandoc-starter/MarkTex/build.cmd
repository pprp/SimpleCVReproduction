python setup.py sdist bdist_wheel
SET CountI=0
SET BakDir="./dist/"
for /f   %%a in ('dir %BakDir% /a:-d /B /o:-D') do (
rem echo %%a
SET FileName=%%a
SET CountI=CountI+1
rem echo %CountI%
if %CountI% == 0 goto bakup2
 
)
:bakup2
rem pip install --force-reinstall %BakDir%%FileName%
pip install %BakDir%%FileName%