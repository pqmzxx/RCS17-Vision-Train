# What I learned today?
## **How to use github**
### 如何将文件上传到github
### 首先将远程仓库克隆到本地仓库 使用git clone命令
### 示例：git clone https://github.com/pqmzxx/RCS17-Vision-Train.git
### 克隆完成后，可以使用以下命令检查远程仓库设置：
### git remote -v
### 若输出类似origin  https://github.com/username/repository.git (fetch)
### origin  https://github.com/username/repository.git (push)则为远程连接成功

---

### **误区：何时需要使用 git remote add origin <repository-url>**
### 初始化本地仓库后：
### 如果你使用 git init 命令初始化了一个新的本地 Git 仓库，但还没有将其与远程仓库关联，你需要使用 git remote add 命令添加远程仓库。（即原是在资源管理器新建的文件，不是在GitHub直接建立具有链接的情况）

***

### 与远程连接成功后
### 我希望在一个新的分支上进行工作，因此先创建分支，然后在新分支上创建文件
### 在创建新的分支之前，可以先查看当前的分支和其他分支：git branch
### 例如：git branch 《new-branch-name》
### 若要创建一个新的分支但不切换到该分支，可以使用：
### git branch baga
### 若要创建并切换到新分支，可以使用：
### git checkout -b baga
### 切换分支：
### git checkout baga
### 在新分支上创建新文件
### 例如：touch test.py
### 下一步将新文件添加到暂存区：
### git add test.py
### 提交新文件到新分支：
### git commit -m "Add test.py"
### 推送新分支到远程仓库：
### git push origin baga
### **若为创建新文件夹与文件只有一些小区别**
### 在新分支上创建一个新文件夹:
### mkdir new-folder
### 你需要在文件夹中创建至少一个文件，才能将文件夹提交到 Git。例如，创建一个 README.md 文件：
### touch new-folder/README.md
### git commit -m "双引号内容自己定均可"
### 推送新分支到远程仓库：
### git push origin baga
***
## 下午主要为cuda等环境配置




