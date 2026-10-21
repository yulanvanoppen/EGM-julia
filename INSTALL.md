# Installing EGM.jl
_Step-by-step installation of EGM.jl, up to a first estimation. For using the package, see [README.md](README.md); for the interactive smoothing app, see [SMOOTHING_APP.md](SMOOTHING_APP.md)_

&nbsp;

## 1. Julia and the REPL
Use **Julia 1.11 or later**. The package itself also runs on Julia 1.10, but the example and benchmark environments need 1.11. If Julia is not installed yet, get it from [julialang.org](https://julialang.org/downloads/). In Julia, `VERSION` shows the installed version.

All commands in this guide are typed in the Julia **REPL** (read–eval–print loop), Julia's interactive prompt. You type a command, Julia runs it and prints the result. Start the REPL by typing `julia` in a terminal, or inside your editor (step 2). The prompt looks like this:
```julia
julia> 1 + 1
2
```
The REPL has a few modes, entered with one key at the start of an empty line, and left again with backspace:
- `]` switches to package mode (`pkg>`), a short form for package commands; e.g. `pkg> status` lists the packages of the active environment;
- `?` switches to help mode, e.g. `help?> egm` shows the documentation of `egm`.

`exit()` closes the REPL. Everything defined in a session (variables, loaded packages) is lost when the REPL is closed or restarted.

## 2. Install an editor (optional)
Any text editor works. For an integrated REPL, plot pane and file execution, install [VS Code](https://code.visualstudio.com/) or [VSCodium](https://vscodium.com/), and then the **Julia** extension from the editor's extension panel. Start a REPL inside the editor with Ctrl+Shift+P → "Julia: Start REPL".

## 3. Get the code
EGM.jl is meant to be used from its source code, so that you can add model files, write parsers or adapt any part of the method. Download the repository into a folder of your choice, either by cloning it with git:
```
git clone https://github.com/yulanvanoppen/EGM-julia
```
or by downloading it as a ZIP file from GitHub ("Code" → "Download ZIP") and unpacking it. With git, you can later update with `git pull` and keep track of your own changes.

The folder (called `EGM-julia/` below) contains the code (`src/`), tests (`test/`), example scripts (`examples/`), benchmark scripts (`benchmarks/`) and the documentation.

## 4. Set up environments
Julia installs packages into _environments_. An environment is a folder whose `Project.toml` lists the packages of a project, so that different projects don't interfere with each other. Packages are downloaded and compiled once per environment; this takes several minutes the first time, and GLMakie (for the smoothing app) adds a few more. Use forward slashes in paths, also on Windows.

### Your own work
Create a folder for your project, e.g. `MyEGMProject`, make it an environment, and link it to your copy of EGM.jl with `develop`:
```julia
julia> using Pkg
julia> Pkg.activate("C:/Users/me/Documents/MyEGMProject")       # creates the environment
julia> Pkg.develop(path = "C:/Users/me/Documents/EGM-julia")      # use the code in this folder
julia> Pkg.add("GLMakie")                                         # optional: only for the smoothing app
```
`develop` makes the environment use the code in `EGM-julia/` directly, so your changes to the code are used in your project.

Next time, activate the environment again before using EGM.jl. Either run `Pkg.activate("C:/Users/me/Documents/MyEGMProject")`, or start Julia in that folder with `julia --project=.`. In VS Code/VSCodium, open the folder (File → Open Folder) and select it as the Julia environment in the status bar.

To have EGM.jl available in every session without activating an environment, run the `develop` command once in your default environment (the one active when Julia starts, shown as `(@v1.11) pkg>` in package mode) instead.

### Environments in the repository
The repository contains three environments of its own, all using the code in `EGM-julia/`:

| Folder | Environment for |
|---|---|
| `EGM-julia/` | the package itself and its tests |
| `EGM-julia/examples/` | the example scripts (includes GLMakie) |
| `EGM-julia/benchmarks/` | the benchmark scripts (see [benchmarks/README.md](benchmarks/README.md)) |

Set up the one(s) you need once, e.g. for the examples:
```julia
julia> Pkg.activate("C:/Users/me/Documents/EGM-julia/examples")
julia> Pkg.instantiate()                                          # downloads and compiles the dependencies
```

To run the tests, activate the package environment and run them (this takes a few minutes):
```julia
julia> Pkg.activate("C:/Users/me/Documents/EGM-julia")
julia> Pkg.test()                        # or Pkg.test(test_args = ["8"]) for test_phase8.jl only
```

## 5. Check the installation
With your environment active:
```julia
julia> using EGM
julia> pathof(EGM), pkgversion(EGM)
```
`pathof` should point into your `EGM-julia/src/` folder, and `pkgversion` should match the `version` in `EGM-julia/Project.toml`.

## 6. Run a first example
With your environment (or the `examples` environment) active:
```julia
julia> include("C:/Users/me/Documents/EGM-julia/examples/example.jl")
```
This simulates noisy measurements of one state of the Higgins–Selkov oscillator, and recovers its parameters and the initial condition of the unobserved state with EGM and with trajectory matching. The smoothing-app examples (`smoothing_app.jl`, `smoothing_app_TCS.jl`) also need GLMakie, which the `examples` environment includes.

In VS Code/VSCodium, you can also open a script and run it with Ctrl+Shift+P → "Julia: Execute File in REPL". Note that Ctrl+Enter runs only the current line or block; select all (Ctrl+A) first to run a whole script.

&nbsp;

## Changing the code
Edit the files in `EGM-julia/` directly, e.g. add a model file or a function in `src/`. Changes take effect in a new Julia session, or immediately if [Revise.jl](https://github.com/timholy/Revise.jl) is loaded before EGM:
```julia
julia> using Revise, EGM
```
Run the tests after changing `src/` (step 4). New functions in `src/` are only visible outside the package after you `export` them in `src/EGM.jl`, or when called as `EGM.name`.

## Updating
- **With git:** `git pull` in `EGM-julia/` (or use your editor's git panel), then restart Julia. If you changed files yourself, commit them first (or keep them on a separate branch), so git can merge the updates.
- **With a ZIP download:** download and unpack the new version, and copy your own changes into it.

## Troubleshooting
- **`git clone` asks for a GitHub login and then reports "not found".** The repository URL is wrong, or the repository is private. For a private repository, enter a GitHub personal access token as the password.
- **"Package X not found in current path".** The wrong environment is active. Check with `Base.active_project()` and activate the right one (step 4).
- **An older version of EGM is loaded,** or a recently added function is "not defined". Another environment holds a different copy of EGM (for example one installed with `Pkg.add`), and it was loaded first. Check `pathof(EGM)`, restart Julia, and activate the intended environment first. Replace the other copy with your folder: activate that environment and run `Pkg.rm("EGM")`, then `Pkg.develop(path = …)`.
- **Precompilation errors in packages unrelated to EGM** (e.g. PyCall). These come from other packages in the same environment. They don't affect EGM, but are best avoided by using a separate environment for your EGM work.
- **The smoothing app reports "needs GLMakie".** Add GLMakie to the active environment and run `using GLMakie` before calling `smoothing_app`.
