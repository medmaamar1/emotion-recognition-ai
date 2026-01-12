"""
Knowledge distillation for model compression and transfer learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """
    Knowledge distillation loss combining hard and soft targets.
    
    Args:
        temperature: Temperature for softening probabilities (default: 4.0)
        alpha: Weight for distillation loss (default: 0.7)
    """
    def __init__(self, temperature=4.0, alpha=0.7):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
    
    def forward(self, student_logits, teacher_logits, labels):
        """
        Compute distillation loss.
        
        Args:
            student_logits: Student model predictions
            teacher_logits: Teacher model predictions
            labels: Ground truth labels
        
        Returns:
            Combined distillation loss
        """
        # Soft targets from teacher
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        
        # KL divergence loss (distillation)
        kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        kd_loss *= (self.temperature ** 2)
        
        # Cross-entropy loss with hard labels
        ce_loss = F.cross_entropy(student_logits, labels)
        
        # Combined loss
        total_loss = self.alpha * kd_loss + (1 - self.alpha) * ce_loss
        
        return total_loss, kd_loss, ce_loss


class MultiTaskDistillationLoss(nn.Module):
    """
    Knowledge distillation loss for multi-task models.
    
    Args:
        temperature: Temperature for softening probabilities (default: 4.0)
        alpha: Weight for distillation loss (default: 0.7)
        task_weights: Weights for each task (default: None)
    """
    def __init__(self, temperature=4.0, alpha=0.7, task_weights=None):
        super(MultiTaskDistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.task_weights = task_weights
    
    def forward(self, student_outputs, teacher_outputs, labels):
        """
        Compute multi-task distillation loss.
        
        Args:
            student_outputs: Tuple of (emotion_logits, au_logits) from student
            teacher_outputs: Tuple of (emotion_logits, au_logits) from teacher
            labels: Tuple of (emotion_labels, au_labels)
        
        Returns:
            Combined distillation loss
        """
        student_emotion, student_au = student_outputs
        teacher_emotion, teacher_au = teacher_outputs
        emotion_labels, au_labels = labels
        
        # Emotion distillation
        teacher_emotion_probs = F.softmax(teacher_emotion / self.temperature, dim=1)
        student_emotion_log_probs = F.log_softmax(student_emotion / self.temperature, dim=1)
        emotion_kd_loss = F.kl_div(student_emotion_log_probs, teacher_emotion_probs, reduction='batchmean')
        emotion_kd_loss *= (self.temperature ** 2)
        
        emotion_ce_loss = F.cross_entropy(student_emotion, emotion_labels)
        
        # AU distillation
        teacher_au_probs = torch.sigmoid(teacher_au)
        student_au_probs = torch.sigmoid(student_au)
        au_mse_loss = F.mse_loss(student_au_probs, teacher_au_probs)
        au_bce_loss = F.binary_cross_entropy_with_logits(student_au, au_labels.float())
        
        # Combine losses
        if self.task_weights is None:
            self.task_weights = [1.0, 1.0]
        
        emotion_loss = self.alpha * emotion_kd_loss + (1 - self.alpha) * emotion_ce_loss
        au_loss = self.alpha * au_mse_loss + (1 - self.alpha) * au_bce_loss
        
        total_loss = self.task_weights[0] * emotion_loss + self.task_weights[1] * au_loss
        
        return total_loss, emotion_loss, au_loss


class FeatureDistillationLoss(nn.Module):
    """
    Feature-level knowledge distillation loss.
    
    Args:
        temperature: Temperature for softening probabilities (default: 4.0)
        alpha: Weight for distillation loss (default: 0.7)
        beta: Weight for feature distillation loss (default: 0.3)
    """
    def __init__(self, temperature=4.0, alpha=0.7, beta=0.3):
        super(FeatureDistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, student_logits, teacher_logits, labels, 
                student_features, teacher_features):
        """
        Compute feature-level distillation loss.
        
        Args:
            student_logits: Student model predictions
            teacher_logits: Teacher model predictions
            labels: Ground truth labels
            student_features: Student model features
            teacher_features: Teacher model features
        
        Returns:
            Combined distillation loss
        """
        # Output-level distillation
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        kd_loss *= (self.temperature ** 2)
        
        ce_loss = F.cross_entropy(student_logits, labels)
        
        # Feature-level distillation
        feature_loss = F.mse_loss(student_features, teacher_features)
        
        # Combined loss
        total_loss = self.alpha * (self.alpha * kd_loss + (1 - self.alpha) * ce_loss) + self.beta * feature_loss
        
        return total_loss, kd_loss, ce_loss, feature_loss


class TeacherStudent(nn.Module):
    """
    Wrapper for teacher-student distillation.
    
    Args:
        teacher: Teacher model
        student: Student model
    """
    def __init__(self, teacher, student):
        super(TeacherStudent, self).__init__()
        self.teacher = teacher
        self.student = student
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
    
    def forward(self, x, return_teacher_output=False):
        """
        Forward pass through both models.
        
        Args:
            x: Input images
            return_teacher_output: Whether to return teacher output (default: False)
        
        Returns:
            Student output, and optionally teacher output
        """
        with torch.no_grad():
            teacher_output = self.teacher(x)
        
        student_output = self.student(x)
        
        if return_teacher_output:
            return student_output, teacher_output
        else:
            return student_output
    
    def train_student(self, mode=True):
        """Set student to train mode while keeping teacher in eval mode."""
        self.student.train(mode)
        self.teacher.eval()
    
    def eval(self):
        """Set both models to eval mode."""
        self.student.eval()
        self.teacher.eval()


class ProgressiveDistillation(nn.Module):
    """
    Progressive knowledge distillation with curriculum learning.
    
    Args:
        teacher: Teacher model
        student: Student model
        total_steps: Total training steps (default: 10000)
        start_alpha: Initial distillation weight (default: 0.1)
        end_alpha: Final distillation weight (default: 0.9)
    """
    def __init__(self, teacher, student, total_steps=10000, 
                 start_alpha=0.1, end_alpha=0.9):
        super(ProgressiveDistillation, self).__init__()
        self.teacher = teacher
        self.student = student
        self.total_steps = total_steps
        self.start_alpha = start_alpha
        self.end_alpha = end_alpha
        self.current_step = 0
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
    
    def get_alpha(self):
        """Get current distillation weight."""
        progress = min(self.current_step / self.total_steps, 1.0)
        return self.start_alpha + (self.end_alpha - self.start_alpha) * progress
    
    def forward(self, x):
        """Forward pass through both models."""
        with torch.no_grad():
            teacher_output = self.teacher(x)
        
        student_output = self.student(x)
        
        return student_output, teacher_output
    
    def step(self):
        """Advance to next step."""
        self.current_step += 1
    
    def train_student(self, mode=True):
        """Set student to train mode."""
        self.student.train(mode)


class EnsembleDistillation(nn.Module):
    """
    Distillation from an ensemble of teachers to a student.
    
    Args:
        teachers: List of teacher models
        student: Student model
        aggregation: Method to aggregate teacher outputs ('mean', 'vote')
    """
    def __init__(self, teachers, student, aggregation='mean'):
        super(EnsembleDistillation, self).__init__()
        self.teachers = nn.ModuleList(teachers)
        self.student = student
        self.aggregation = aggregation
        
        # Freeze teachers
        for teacher in self.teachers:
            for param in teacher.parameters():
                param.requires_grad = False
            teacher.eval()
    
    def forward(self, x):
        """Forward pass through teachers and student."""
        # Get teacher outputs
        teacher_outputs = []
        with torch.no_grad():
            for teacher in self.teachers:
                teacher_outputs.append(teacher(x))
        
        # Aggregate teacher outputs
        if self.aggregation == 'mean':
            teacher_output = torch.mean(torch.stack(teacher_outputs), dim=0)
        elif self.aggregation == 'vote':
            teacher_probs = [torch.softmax(t, dim=1) for t in teacher_outputs]
            teacher_output = torch.mean(torch.stack(teacher_probs), dim=0)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        
        # Get student output
        student_output = self.student(x)
        
        return student_output, teacher_output
    
    def train_student(self, mode=True):
        """Set student to train mode."""
        self.student.train(mode)


def distill_model(teacher, student, train_loader, val_loader, device, 
                  epochs=50, lr=0.001, temperature=4.0, alpha=0.7):
    """
    Perform knowledge distillation from teacher to student.
    
    Args:
        teacher: Teacher model
        student: Student model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        epochs: Number of training epochs (default: 50)
        lr: Learning rate (default: 0.001)
        temperature: Distillation temperature (default: 4.0)
        alpha: Distillation weight (default: 0.7)
    
    Returns:
        Trained student model
    """
    # Create teacher-student wrapper
    teacher_student = TeacherStudent(teacher, student).to(device)
    
    # Loss function
    criterion = DistillationLoss(temperature=temperature, alpha=alpha)
    
    # Optimizer
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # Training
        teacher_student.train_student()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            student_output, teacher_output = teacher_student(images, return_teacher_output=True)
            
            # Compute loss
            loss, kd_loss, ce_loss = criterion(student_output, teacher_output, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(student_output, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        train_loss /= len(train_loader)
        
        # Validation
        teacher_student.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                
                student_output = teacher_student.student(images)
                loss = F.cross_entropy(student_output, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(student_output, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(student.state_dict(), 'best_student_model.pth')
    
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    return student


if __name__ == "__main__":
    # Test distillation utilities
    print("Testing knowledge distillation utilities...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 4
    num_classes = 14
    
    # Create dummy models
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(16, num_classes)
        
        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    teacher = DummyModel().to(device)
    student = DummyModel().to(device)
    
    # Test DistillationLoss
    print("\nTesting DistillationLoss...")
    criterion = DistillationLoss(temperature=4.0, alpha=0.7)
    
    dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
    teacher_output = teacher(dummy_input)
    student_output = student(dummy_input)
    labels = torch.randint(0, num_classes, (batch_size,)).to(device)
    
    total_loss, kd_loss, ce_loss = criterion(student_output, teacher_output, labels)
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"KD loss: {kd_loss.item():.4f}")
    print(f"CE loss: {ce_loss.item():.4f}")
    
    # Test TeacherStudent wrapper
    print("\nTesting TeacherStudent wrapper...")
    teacher_student = TeacherStudent(teacher, student)
    student_output = teacher_student(dummy_input)
    print(f"Student output shape: {student_output.shape}")
    
    student_output, teacher_output = teacher_student(dummy_input, return_teacher_output=True)
    print(f"Student output shape: {student_output.shape}")
    print(f"Teacher output shape: {teacher_output.shape}")
    
    # Test ProgressiveDistillation
    print("\nTesting ProgressiveDistillation...")
    progressive = ProgressiveDistillation(teacher, student, total_steps=1000)
    for step in [0, 250, 500, 750, 1000]:
        progressive.current_step = step
        alpha = progressive.get_alpha()
        print(f"Step {step}: alpha = {alpha:.3f}")
    
    # Test EnsembleDistillation
    print("\nTesting EnsembleDistillation...")
    teachers = [DummyModel().to(device), DummyModel().to(device)]
    ensemble_distill = EnsembleDistillation(teachers, student, aggregation='mean')
    student_output, teacher_output = ensemble_distill(dummy_input)
    print(f"Student output shape: {student_output.shape}")
    print(f"Teacher output shape: {teacher_output.shape}")
    
    print("\nAll distillation utilities tested successfully!")
