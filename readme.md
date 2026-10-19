# Jupyter setup

The project uses the existing `.venv` virtual environment. To activate it in
PowerShell and start Jupyter Notebook:

```powershell
.\.venv\Scripts\Activate.ps1
python -m notebook
```

Install the project dependencies, including Jupyter, with:

```powershell
python -m pip install -r requirements.txt
```