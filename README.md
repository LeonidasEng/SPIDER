<a name="readme-top"></a>

<!-- PROJECT SHIELDS -->
<!--
*** Project Shields can go here if the project is publically available.
-->

<!-- PROJECT LOGO -->
<div align="center">
    <img src="docs/SPIDER_Mission_Logo.png" alt="Logo", width="500">
  <br />
  
  <p align="center">
    Data-driven ML framework for quantifying the reliability and uncertainty of space weather forecasts 
    and identifying when they can be trusted.
    <br />
    <a href="https://github.com/LeonidasEng/SPIDER/issues">Report Bug</a>
    ·
    <a href="https://github.com/LeonidasEng/SPIDER/issues">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#initial-setup">Initial Setup</a></li>
        <ul>
        <li><a href="#️-setting-spider-environment-variable---required">Setting SPIDER Environment Variable - Required</a></li>
            <ul>
            <li><a href="#method-1-gui">Method 1 GUI</a></li>
            <li><a href="#method-2-command-line">Method 2 Command Line</a></li>
            </ul>
        </a></li>
        <li><a href="#️-python-virtual-environment-setup---recommended">Python Virtual Environment - Recommended</a></li>
        <li><a href="#troubleshooting">Troubleshooting</a></li>
        </ul>
    <li><a href="#user-guide">User Guide</a></li>
    <li><a href="#contributing">Contributing</a></li>
        <ul>
        <li><a href="#clone-the-repo">Clone the Repo</a></li>
        <li><a href="#suggestions">Suggestions</a></li>
        </ul>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## Initial Setup 
PLEASE REVIEW BEFORE RUNNING ANY SCRIPTS

### ⚠️ SETTING SPIDER ENVIRONMENT VARIABLE - REQUIRED 

The SPIDER environment variable must point to the root directory of the SPIDER project. This allows scripts to locate data, models, and configuration files.

- WINDOWS Guide

<details>
<summary>Click to expand/collapse</summary>

<b>Method 1: GUI</b>
<details>
<summary>Click to expand/collapse</summary>

<ol>
<li>Open the Start menu and search for "Environment Variables".</li>
<li>Click "Edit the system environment variables".</li>
<li>In the System Properties window, click "Environment Variables...".</li>
<li>Under "User variables", click "New...".</li>
<li>Enter:<br>
Variable name: SPIDER<br>
Variable value: C:\path\to\SPIDER</li>
<li>Click OK to save.</li>
</ol>

<p>Restart any open terminals or IDEs.</p>

<p><b>Verify:</b></p>

<p>Command Prompt:</p>
<pre><code class="language-sh">echo %SPIDER%</code></pre>

<p>PowerShell:</p>
<pre><code class="language-sh">echo $env:SPIDER</code></pre>

</details>

<b>Method 2: Command Line</b>
<details>
<summary>Click to expand/collapse</summary>

<p><b>Temporary (current session only):</b></p>

<p>Command Prompt:</p>
<pre><code class="language-sh">set SPIDER=C:\path\to\SPIDER</code></pre>

<p>PowerShell:</p>
<pre><code class="language-sh">$env:SPIDER = "C:\path\to\SPIDER"</code></pre>

<p><b>Permanent (user-level):</b></p>

<p>Command Prompt:</p>
<pre><code class="language-sh">setx SPIDER "C:\path\to\SPIDER"</code></pre>

<p>Restart the terminal after setting.</p>

</details>

</details>

- LINUX Guide
<details>
<summary>Click to expand/collapse</summary>

<p><b>Temporary (current session only):</b></p>

<pre><code class="language-sh">export SPIDER="/path/to/SPIDER"</code></pre>

<p><b>Verify:</b></p>

<pre><code class="language-sh">echo $SPIDER</code></pre>

<p><b>Permanent (user-level):</b></p>

<p>Open your shell configuration file:</p>

<p>Bash:</p>
<pre><code class="language-sh">nano ~/.bashrc</code></pre>

<p>Add the following line:</p>

<pre><code class="language-sh">export SPIDER="/path/to/SPIDER"</code></pre>

<p>Reload the configuration:</p>

<pre><code class="language-sh">source ~/.bashrc</code></pre>

</details>

### ⚠️ PYTHON VIRTUAL ENVIRONMENT SETUP - RECOMMENDED

This project uses a Python virtual environment to isolate dependencies from other projects. It is recommended that you create
one.
<details>
<summary>Click to expand/collapse</summary>

<p>To create a virtual environment:</p>
<pre><code class="language-sh">python -m venv venv</code></pre>

<p>Activate the environment:</p>

<p><b>Windows:</b></p>
<pre><code class="language-sh">venv\Scripts\activate</code></pre>

<p><b>Linux/macOS:</b></p>
<pre><code class="language-sh">source venv/bin/activate</code></pre>

<p>Install required packages:</p>
<pre><code class="language-sh">pip install -r requirements.txt</code></pre>

</details>

#### Troubleshooting 
<details>
<summary>Click to expand/collapse</summary>

<p><b>Missing Python Packages</b></p>

<p>If a package is installed but your script cannot find it.</p>

<p>Check which Python is running:</p>
<pre><code class="language-sh">python -c "import sys; print(sys.executable)"</code></pre>

<p>Make sure it points to your venv (e.g. venv\Scripts\python.exe)</p>

<p>Check where the package is installed:</p>
<pre><code class="language-sh">pip show &lt;package_name&gt;</code></pre>

<p>Ensure the Location is inside your venv:<br>
...\venv\Lib\site-packages</p>

<p>If it shows AppData or another path, it is installed in the wrong environment.</p>

<p>Install into the correct environment:</p>
<pre><code class="language-sh">python -m pip install &lt;package_name&gt;</code></pre>

<p>Verify:</p>
<pre><code class="language-sh">python -c "import &lt;package_name&gt;"</code></pre>

<p><b>KeyboardInterrupt cancels debug session</b></p>

<p>In Windows VSCode, if the venv keeps closing on every debug session, try adding the following line to .vscode/settings.json:</p>
<p><code class="language-sh">{"python.useEnvironmentsExtension": true}</code></p>
<p><a href="https://github.com/microsoft/vscode-python/issues/25720">Bug Report</a></p>
</details>
<p align="right">(<a href="#readme-top">back to top</a>)</p>


## User Guide
Welcome to the repo! In order to start using SPIDER, first ensure you have read the
"Initial Setup" guide before attempting to run any script. 

### Data Collection
<details>
<summary>Click to expand/collapse</summary>
<p>To download new data, run:</p>
<pre><code class="language-sh">python -m src.ingestion.ftp_access</code></pre>

<p>This opens the FTP Access utility main menu.</p>

<p>From the menu, select the data source you want to use, for example <code>forecasts</code>.</p>

<p>Next, select the data you want to download, for example <code>3day</code>.</p>

<p>The utility will connect to the FTP server and display the available data.</p>

<p>Enter a start date in <code>YYYYMMDD</code> format, for example <code>20250101</code>.</p>

<p>Enter an end date in <code>YYYYMMDD</code> format, for example <code>20250131</code>.</p> 

<p>After you press Enter, the selected data will be downloaded to:</p> 
<pre><code>data/raw/&lt;start_date&gt;_&lt;end_date&gt;_raw/&lt;year&gt;/&lt;month&gt;</code></pre>

</details>

### Data Parsing
<details>
<summary>Click to expand/collapse</summary>
<p>All raw data collected via FTP access utility will be stored in:</p>
<pre><code>data/raw/&lt;type&gt;/&lt;subtype&gt;</code></pre>

<p>To process the new data, run:</p>
<pre><code class="language-sh">python -m src.ingestion.parse_3day</code></pre>

<p>The parser will automatically locate and process all files</p>

<p>Recursive conflict-handling will ensure only the most recent data is used.</p>

</details>

### Build Datasets
<details>
<summary>Click to expand/collapse</summary>

</details>

### ML Modelling
<details>
<summary>Click to expand/collapse</summary>

</details>

### Rule Layer
<details>
<summary>Click to expand/collapse</summary>

</details>

## ℹ️ Data Disclaimer
NOAA provided historical archives for 3-day Kp forecast data period (2012-2024) to resolve data gaps.
However, NOAA states that the data collected using this tool is not definitive and may still contain 
errors. It is a best-effort attempt to fill in gaps in the archive.

SPIDER currently can only gather data on historic forecasts. The main limitation being that observed Kp
is not available at forecast runtime. 

NASAs OMNI2 uses placeholder values to indicate missing or invalid data. OMNI2 

## Built With
* Python
* Scikit-learn

<!-- CHANGELOG -->
<!--
## Changelog
You can find the changelog for this repo here: [Changelog](https://github.com/LeonidasEng/SPIDER/tree/trunk/CHANGELOG.md)
-->

See the [open issues](https://github.com/LeonidasEng/SPIDER/issues) for a list of known issues.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing
This project was created as a Final year Engineering project. Any feedback you can give is **greatly appreciated**.

### Clone the Repo
1. Clone the repository to your local machine or download the ZIP.
2. On the repository page, click on the green "Code" button.
3. Copy the HTTPS or SSH URL of the repository (you can toggle between the two options).
4. Open a terminal or command prompt on your local machine.
5. Navigate to the directory where you want to clone the repository.
6. Run the following command for HTTPS:
    ```
    git clone https://github.com/LeonidasEng/SPIDER.git
    ```

### Suggestions
If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/NewFeature`)
3. Commit your Changes (`git commit -m 'Add some NewFeature'`)
4. Push to the Branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License
This source code is licensed under the Apache2.0-style license found in: [LICENSE](https://github.com/LeonidasEng/SPIDER/blob/trunk/LICENSE) found in the root directory of this source tree.

Project Link: [https://github.com/LeonidasEng/SPIDER](https://github.com/LeonidasEng/SPIDER)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments
* [NOAA Space Weather Prediction Center](https://www.swpc.noaa.gov/)
* [NOAA National Center for Environmental Information](https://www.ngdc.noaa.gov/stp/)
* [NASA Goddard Space Flight Center](https://omniweb.gsfc.nasa.gov/)
* [Best-README-Template](https://github.com/othneildrew/Best-README-Template)


<p align="right">(<a href="#readme-top">back to top</a>)</p>





<!-- MARKDOWN LINKS & IMAGES -->
<!--
*** Shields could be added when project is public:
*** [contributors-shield]: 
*** [contributors-url]: 
*** [forks-shield]: 
*** [forks-url]:
*** [stars-shield]: 
*** [stars-url]: 
*** [issues-shield]: 
*** [issues-url]: 
*** [license-shield]: 
*** [license-url]: 
*** [linkedin-shield]: 
*** [linkedin-url]: 
*** [project-banner]: 

-->