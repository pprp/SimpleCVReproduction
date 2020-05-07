set /p commitmsg=input commitmsg:
git add *
git commit -m %commitmsg%
git push