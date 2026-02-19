# AI Agents Internship Projects

This repository contains AI agent projects developed by our internship team.

## 📁 Repository Structure

Each intern has their own folder containing their projects:

```
.
├── Agents/           # [Your Name] - Voice Assistant & Email Agent Projects
├── [intern2_name]/   # [Intern 2]'s projects
├── [intern3_name]/   # [Intern 3]'s projects
└── ...
```

## 🚀 Getting Started

Each project folder contains:
- Source code files
- `requirements.txt` or `requirements_*.txt` for dependencies
- Project-specific README with setup instructions

## 📋 Contributing Guidelines

### For Interns Adding New Projects:

1. **Create Your Folder**
   ```bash
   mkdir YourName
   cd YourName
   ```

2. **Add Your Project Files**
   - Include all source code
   - Add `requirements.txt` with all dependencies
   - Create a README.md explaining your project

3. **Create .env.example**
   - Never commit actual `.env` files with API keys
   - Provide an example template instead

4. **Commit Your Changes**
   ```bash
   git add YourName/
   git commit -m "Add [YourName]: [Brief project description]"
   git push origin main
   ```

### Best Practices:

- ✅ Write clear commit messages
- ✅ Document your code
- ✅ Include setup instructions in your README
- ✅ Test your code before committing
- ❌ Never commit API keys or passwords
- ❌ Don't commit `__pycache__` or virtual environments

## 🔒 Security

**Important:** This repository uses `.gitignore` to prevent committing sensitive files like:
- `.env` files containing API keys
- `__pycache__` directories
- Virtual environment folders

Always use `.env.example` files to show what environment variables are needed.

## 📞 Contact

For questions or issues, contact [Senior's Name/Email]

---
*Last Updated: February 2026*
