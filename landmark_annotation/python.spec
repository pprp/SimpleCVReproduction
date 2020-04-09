# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['keypoints.pyD:\\Programs\\miniconda3\\envs\\clean\\python.exe', 'd:/Github/SimpleCVReproduction/landmark_annotation/keypoints.py'],
             pathex=['D:\\Github\\SimpleCVReproduction\\landmark_annotation'],
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='python',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=False , icon='feather.ico')
